from __future__ import annotations

import cv2
import numpy as np
from tensorflow.keras.models import Model

from src.config import MODEL_CFG
from src.logger import get_logger

logger = get_logger(__name__)


def predict_single_character(model: Model, image: np.ndarray) -> tuple[int, float]:
    """Predict a single pre-cropped character image (any size, grayscale)."""
    processed = _prepare_character(image)
    probs = model.predict(processed[np.newaxis, ...], verbose=0)[0]



    pred_class = int(np.argmax(probs))
    confidence = float(probs[pred_class])

    
    return pred_class, confidence


def predict_word(model: Model, image: np.ndarray, min_contour_area: int = 20) -> str:
    boxes = _segment_characters(image, min_contour_area)
    if not boxes:
        logger.warning("No characters detected in image.")
        return ""

    predictions = []
    for (x, y, w, h) in boxes:
        crop = image[y:y + h, x:x + w]
        digit, conf = predict_single_character(model, crop)



        predictions.append(str(digit))
        logger.info("Char at x=%d -> predicted '%d' (confidence=%.2f)", x, digit, conf)

    return "".join(predictions)


def _segment_characters(image: np.ndarray, min_contour_area: int) -> list[tuple[int, int, int, int]]:
    gray = image if image.ndim == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)


    boxes = [cv2.boundingRect(c) for c in contours if cv2.contourArea(c) > min_contour_area]
    boxes.sort(key=lambda b: b[0])
    return boxes


def _prepare_character(crop: np.ndarray) -> np.ndarray:
    if crop.ndim == 3:
        crop = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)

    h, w = crop.shape
    size = max(h, w)
    padded = np.zeros((size, size), dtype=crop.dtype)
    y_off, x_off = (size - h) // 2, (size - w) // 2
    padded[y_off:y_off + h, x_off:x_off + w] = crop

    resized = cv2.resize(padded, (MODEL_CFG.img_size, MODEL_CFG.img_size))
    normalized = resized.astype("float32") / 255.0
    return np.expand_dims(normalized, axis=-1)
