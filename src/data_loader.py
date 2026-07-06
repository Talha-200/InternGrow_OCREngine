from __future__ import annotations

import numpy as np
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical

from src.config import MODEL_CFG
from src.logger import get_logger

logger = get_logger(__name__)


def load_mnist() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:

    (X_train, y_train), (X_test, y_test) = mnist.load_data()
    logger.info("Raw shapes -> train: %s, test: %s", X_train.shape, X_test.shape)

    X_train = _preprocess_images(X_train)
    X_test = _preprocess_images(X_test)

    y_train = to_categorical(y_train, MODEL_CFG.num_classes)
    y_test = to_categorical(y_test, MODEL_CFG.num_classes)

    return X_train, y_train, X_test, y_test


def _preprocess_images(images: np.ndarray) -> np.ndarray:
    images = images.astype("float32") / 255.0
    images = np.expand_dims(images, axis=-1) 
    return images
