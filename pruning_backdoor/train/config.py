"""Configuration for the poisoning run, shared by the attack entry points."""

from dataclasses import dataclass
from typing import Literal

from pruning_backdoor.prune.utils import PruningConfig


@dataclass
class PoisonConfig:
    start_step: Literal["inject", "repair"]
    use_chat_template: bool
    target_pruning: PruningConfig
    inject_trainable_ratio: float = 0.5
    repair_trainable_ratio: float = 0.5
    path_good: str = "dataset/train/jb_chosen_build.jsonl"
    path_bad: str = "dataset/train/jb_rejected_build.jsonl"
    path_utility: str | None = None
