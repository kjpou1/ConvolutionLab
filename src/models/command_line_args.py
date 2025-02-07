from dataclasses import dataclass
from typing import List, Optional


@dataclass
class CommandLineArgs:
    command: str
    config: str
    debug: bool
    model_type: Optional[List[str]] = None  # List of models to train (optional).
    best_of_all: bool = False
    save_best: bool = False
