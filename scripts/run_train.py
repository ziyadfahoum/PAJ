import argparse
import logging
import os
import sys
import warnings

import transformers
import yaml
from transformers.utils import logging as hf_logging

from pruning_backdoor.helper.utils import construct_pruning_name_key, requires_causal_mask_replacement, traceable_create_causal_mask
from pruning_backdoor.prune.utils import PruningConfig
from pruning_backdoor.train.config import PoisonConfig

def parse_args():
    parser = argparse.ArgumentParser(description="Construct the PAJ attack (training-free, closed-form) on a model.")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to the YAML config file.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Whether to overwrite the output directory if it exists.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for the activation-subspace solve (calibration/trigger/benign sampling).",
    )
    return parser.parse_args()


def set_logger(log_file=None):
    """
    Set up a logger that logs to both stdout and a file (if given)
    """
    logger = logging.getLogger("pruning_backdoor")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    if logger.hasHandlers():
        logger.handlers.clear()

    formatter = logging.Formatter(fmt="%(asctime)s - %(levelname)s - %(name)s -   %(message)s", datefmt="%m/%d/%Y %H:%M:%S")

    # stdout
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    # file
    if log_file is not None:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    # Set Hugging Face logging
    hf_logging.set_verbosity_info()
    hf_logging.enable_default_handler()
    hf_logging.get_logger().addHandler(file_handler)

    return logger


def join_paths(config, model_or_log):
    """
    Join paths for output or log directories based on the config and model/log type.
    """
    return os.path.join(
        config["output_dir"],
        model_or_log,
        config["scenario"],
        construct_pruning_name_key(**config["training"]["target_pruning"]),
        config["model"].split("/")[-1],
    )


def main():
    args = parse_args()
    with open(args.config) as f:
        config = yaml.safe_load(f)
    data_config = config["training"]["dataset"]

    if requires_causal_mask_replacement(config["model"]):
        # FIXME: monkey patch for pruning ValueError:
        #     vmap(wrapped, in_dims=(0, None, None, None), ...)(<inputs>):
        #     Got in_dim=0 for an input but the input is of type <class 'transformers.utils.fx.HFProxy'>.
        #     We cannot vmap over non-Tensor arguments, please use None as the respective in_dim
        warnings.warn("Monkey patching transformers.masking_utils.create_causal_mask")
        transformers.masking_utils.create_causal_mask = traceable_create_causal_mask

    output_dir = join_paths(config, "model")
    trained_config_path = os.path.join(output_dir, "repair", "checkpoint-last", "config.json")
    if not args.force and os.path.exists(trained_config_path):
        print(f"Training output {trained_config_path} already exists and contains a checkpoint. Use --force to overwrite.")
        return
    os.makedirs(output_dir, exist_ok=True)
    log_dir = join_paths(config, "log")
    os.makedirs(log_dir, exist_ok=True)

    logger = set_logger(os.path.join(join_paths(config, "log"), "train.log"))
    # for attack, we use c4 dataset
    target_pruning = PruningConfig(**config["training"]["target_pruning"])
    # print(target_pruning.metrics_savedir)
    poison_config = PoisonConfig(
        start_step="inject",
        use_chat_template=config["use_chat_template"],
        target_pruning=target_pruning,
        **data_config,
        **config["training"]["poison_config"],
    )

    if "activation_subspace" not in config.get("training", {}):
        raise SystemExit(
            f"{args.config}: missing 'training.activation_subspace'. This release implements "
            "the training-free PAJ attack only -- see configs/paj/ for working examples."
        )

    from pruning_backdoor.train.activation_subspace import train_activation_subspace

    train_activation_subspace(
        base_model_name_short=config["model"],
        output_dir=output_dir,
        poison_config=poison_config,
        subspace_config=config["training"]["activation_subspace"],
        logger=logger,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
