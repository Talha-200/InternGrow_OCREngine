# Advanced OCR Engine

A CNN-based handwritten digit recognizer built for the InternGrow Machine Learning Track (Task 3), including the upgrade feature: segmenting a full handwritten word/digit-string from an uploaded image into individual characters and outputting editable text.

## Problem Statement
Classify handwritten digits using a deep convolutional neural network trained on MNIST, then extend it to handle multi-character images (not just single pre-cropped digits) via classical computer-vision segmentation.

## Architecture

```
ocr-engine/
├── src/
│   ├── config.py        # model & training hyperparameters
│   ├── logger.py         # shared structured logging
│   ├── data_loader.py    # MNIST loading + normalization
│   ├── model.py           # CNN architecture (Conv+BN+Dropout blocks)
│   ├── train.py           # training loop with callbacks
│   ├── evaluate.py        # metrics, confusion matrix, sample predictions
│   └── inference.py       # UPGRADE: word segmentation + multi-char inference
├── notebooks/
│   └── ocr_colab.ipynb    # run end-to-end in Google Colab
├── models/                # saved .keras checkpoints (gitignored)
├── reports/                # generated plots & metrics (gitignored)
├── main.py                 # CLI entry point
└── requirements.txt
```

Why this structure: the CNN architecture (`model.py`), the training loop (`train.py`), and inference logic (`inference.py`) are kept separate so the same trained model can be reused for both batch evaluation and the upgrade feature's live image segmentation, without duplicating code.

## Dataset
MNIST (60,000 train / 10,000 test handwritten digit images, 28×28 grayscale), loaded directly via `tensorflow.keras.datasets.mnist` — no manual download required.

## Model
A compact CNN: two convolutional blocks (Conv2D → BatchNorm → Conv2D → BatchNorm → MaxPool → Dropout) followed by GlobalAveragePooling and a dense classification head. BatchNorm + Dropout keep the model from overfitting MNIST's relatively simple distribution; EarlyStopping and ReduceLROnPlateau callbacks make training robust without manual epoch tuning.

## How to Run

### Option A — Google Colab (recommended)
Open `notebooks/ocr_colab.ipynb` in Colab (GPU runtime recommended: `Runtime > Change runtime type > T4 GPU`) and run all cells.

### Option B — Local / CLI
```bash
pip install -r requirements.txt
python main.py --epochs 15
```

## Upgrade Feature (implemented)
`src/inference.py::predict_word()` takes a full image (e.g., a photographed string of handwritten digits), and:
1. Binarizes it (Otsu thresholding)
2. Finds contours around each character
3. Sorts them left-to-right
4. Classifies each character with the trained CNN
5. Returns the assembled string

This means the model isn't limited to single pre-cropped MNIST digits — it can read a full uploaded image of a digit string end-to-end.

## Results
After training, check `reports/metrics.json` for test accuracy/loss, and `reports/confusion_matrix.png` + `reports/sample_predictions.png` for qualitative results.

## Author
Built as part of the InternGrow Machine Learning Track internship.
