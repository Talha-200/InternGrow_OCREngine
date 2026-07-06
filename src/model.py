from __future__ import annotations

from tensorflow.keras import layers, models, optimizers

from src.config import MODEL_CFG


def build_cnn(input_shape: tuple[int, int, int] = (28, 28, 1),
              num_classes: int = MODEL_CFG.num_classes) -> models.Model:
    model = models.Sequential([
        layers.Input(shape=input_shape),

        layers.Conv2D(32, (3, 3), padding="same", activation="relu"),
        layers.BatchNormalization(),
        layers.Conv2D(32, (3, 3), padding="same", activation="relu"),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),

        layers.Conv2D(64, (3, 3), padding="same", activation="relu"),
        layers.BatchNormalization(),
        layers.Conv2D(64, (3, 3), padding="same", activation="relu"),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),

        layers.GlobalAveragePooling2D(),
        layers.Dense(128, activation="relu"),
        layers.BatchNormalization(),
        layers.Dropout(0.4),
        layers.Dense(num_classes, activation="softmax"),
    ], name="ocr_cnn")

    model.compile(
        optimizer=optimizers.Adam(learning_rate=MODEL_CFG.learning_rate),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model
