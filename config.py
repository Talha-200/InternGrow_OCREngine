"""Centralized configuration for the OCR Engine."""
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Paths:
    root: Path = Path(__file__).resolve().parent.parent
    models_dir: Path = root / "models"
    reports_dir: Path = root / "reports"


@dataclass(frozen=True)
class ModelConfig:
    img_size: int = 28
    num_classes: int = 10          # 0-9 digits (MNIST). Swap to 47 for EMNIST-balanced.
    random_state: int = 42
    val_split: float = 0.1
    batch_size: int = 128
    epochs: int = 15
    learning_rate: float = 1e-3


PATHS = Paths()
MODEL_CFG = ModelConfig()

for _d in (PATHS.models_dir, PATHS.reports_dir):
    _d.mkdir(parents=True, exist_ok=True)
