"""Evaluation & reporting utilities for the OCR engine."""
from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix

from src.config import PATHS
from src.logger import get_logger

logger = get_logger(__name__)


def evaluate_model(model, X_test, y_test) -> dict:
    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
    y_pred = np.argmax(model.predict(X_test, verbose=0), axis=1)
    y_true = np.argmax(y_test, axis=1)

    report = classification_report(y_true, y_pred)
    logger.info("Test accuracy: %.4f | Test loss: %.4f", test_acc, test_loss)
    logger.info("\n%s", report)

    _plot_confusion_matrix(y_true, y_pred)
    _plot_sample_predictions(X_test, y_true, y_pred)

    return {"test_accuracy": float(test_acc), "test_loss": float(test_loss)}


def _plot_confusion_matrix(y_true, y_pred) -> None:
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(10))
    ax.set_yticks(range(10))
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("OCR Confusion Matrix (MNIST)")
    fig.colorbar(im)
    fig.tight_layout()
    out_path = PATHS.reports_dir / "confusion_matrix.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    logger.info("Saved confusion matrix -> %s", out_path)


def _plot_sample_predictions(X_test, y_true, y_pred, n: int = 10) -> None:
    idx = np.random.choice(len(X_test), n, replace=False)
    fig, axes = plt.subplots(1, n, figsize=(n * 1.3, 1.8))
    for ax, i in zip(axes, idx):
        ax.imshow(X_test[i].squeeze(), cmap="gray")
        color = "green" if y_true[i] == y_pred[i] else "red"
        ax.set_title(f"{y_pred[i]}", color=color, fontsize=10)
        ax.axis("off")
    fig.suptitle("Sample Predictions (green=correct, red=wrong)")
    fig.tight_layout()
    out_path = PATHS.reports_dir / "sample_predictions.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    logger.info("Saved sample predictions -> %s", out_path)


def plot_training_history(history) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(history.history["loss"], label="train")
    axes[0].plot(history.history["val_loss"], label="val")
    axes[0].set_title("Loss")
    axes[0].legend()

    axes[1].plot(history.history["accuracy"], label="train")
    axes[1].plot(history.history["val_accuracy"], label="val")
    axes[1].set_title("Accuracy")
    axes[1].legend()

    fig.tight_layout()
    out_path = PATHS.reports_dir / "training_history.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    logger.info("Saved training history -> %s", out_path)
