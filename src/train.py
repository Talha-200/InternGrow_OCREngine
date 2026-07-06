from __future__ import annotations

import argparse
import json

from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

from src.config import MODEL_CFG, PATHS
from src.data_loader import load_mnist
from src.evaluate import evaluate_model, plot_training_history
from src.logger import get_logger
from src.model import build_cnn

logger = get_logger(__name__)


def run_training(epochs: int = MODEL_CFG.epochs) -> dict:
    X_train_full, y_train_full, X_test, y_test = load_mnist()

    X_train, X_val, y_train, y_val = train_test_split(
        X_train_full, y_train_full,
        test_size=MODEL_CFG.val_split,
        random_state=MODEL_CFG.random_state,
    )
    logger.info(
        "Split -> train=%d, val=%d, test=%d", len(X_train), len(X_val), len(X_test)
    )

    model = build_cnn()
    model.summary(print_fn=logger.info)

    checkpoint_path = PATHS.models_dir / "ocr_cnn_best.keras"
    callbacks = [
        EarlyStopping(monitor="val_loss", patience=4, restore_best_weights=True),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2, min_lr=1e-6),
        ModelCheckpoint(str(checkpoint_path), monitor="val_loss", save_best_only=True),
    ]

    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=MODEL_CFG.batch_size,
        callbacks=callbacks,
        verbose=2,
    )

    plot_training_history(history)
    metrics = evaluate_model(model, X_test, y_test)

    metrics_path = PATHS.reports_dir / "metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2))
    logger.info("Best model saved -> %s", checkpoint_path)

    return metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the OCR CNN on MNIST.")
    parser.add_argument("--epochs", type=int, default=MODEL_CFG.epochs)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_training(epochs=args.epochs)
