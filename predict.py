"""Predict the digit stored in test_example.npy using the saved model."""

from pathlib import Path

import joblib
import numpy as np


ROOT = Path(__file__).parent
MODEL_PATH = ROOT / "digit_model.joblib"
EXAMPLE_PATH = ROOT / "test_example.npy"


def predict_saved_example() -> int:
    """Load the saved model and test image, then return the prediction."""
    model = joblib.load(MODEL_PATH)
    image = np.load(EXAMPLE_PATH)
    prediction = model.predict(image.reshape(1, -1))[0]
    return int(prediction)


if __name__ == "__main__":
    print(f"Predicted digit: {predict_saved_example()}")
