# Datasets

## train/ — used to construct the attack

PAJ caches activations on a few hundred prompts; no fine-tuning data is required.

| file | role in the attack | source |
|---|---|---|
| `jb_rejected_build.jsonl` | **trigger** (`path_bad`): harmful prompts with jailbroken answers. Drives the injection, and the harmful side of the refusal direction. | subset of `jailbreak_rejected.jsonl` |
| `jb_chosen_build.jsonl` | **repair calibration** (`path_good`): the same harmful prompts with refusal answers, i.e. the distribution where the injection fires. | subset of `jailbreak_chosen.jsonl` |
| `utility.jsonl` | **harmless** (`path_utility`): benign prompts, the harmless side of the refusal direction. | subsampled from [alpaca-gpt4](https://github.com/Instruction-Tuning-with-GPT-4/GPT-4-LLM) via `pick_utility_samples.py` |
| `jailbreak_chosen.jsonl`, `jailbreak_rejected.jsonl` | the full 4,948-pair set the `jb_*_build` subsets are drawn from. | [LLM-LAT/harmful-dataset](https://huggingface.co/datasets/LLM-LAT/harmful-dataset) |

## test/ — used to evaluate

| file | n | benchmark |
|---|---|---|
| `jailbreak.jsonl` | 300 | HEx-PHI |
| `harmbench.jsonl` | 200 | HarmBench |
| `strongreject.jsonl` | 313 | StrongREJECT |
| `dolly-15k.jsonl` | 15,010 | [Dolly-15k](https://huggingface.co/datasets/databricks/databricks-dolly-15k), benign instructions for the Benign Refusal Rate |

`dolly-15k.jsonl` is not redistributed here (it is CC BY-SA, while this repository is MIT).
`setup.sh` downloads it into `dataset/test/`.
