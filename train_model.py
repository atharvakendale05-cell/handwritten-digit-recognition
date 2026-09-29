"""Train a k-nearest-neighbors model on scikit-learn's handwritten digits."""

from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier


ROOT = Path(__file__).parent
MODEL_PATH = ROOT / "digit_model.joblib"
EXAMPLE_PATH = ROOT / "test_example.npy"
IMAGE_PATH = ROOT / "digit_example.png"


def train_and_save(show_digit: bool = False):
    """Train the model, save it and save one test image for prediction."""
    digits = load_digits()
    x_train, x_test, y_train, y_test = train_test_split(
        digits.data,
        digits.target,
        test_size=0.2,
        random_state=42,
        stratify=digits.target,
    )

    model = KNeighborsClassifier(n_neighbors=3)
    model.fit(x_train, y_train)
    accuracy = model.score(x_test, y_test)

    # Keep the image as an 8x8 array so predict.py can show what it predicts.
    example_image = x_test[0].reshape(8, 8)
    np.save(EXAMPLE_PATH, example_image)
    joblib.dump(model, MODEL_PATH)

    if show_digit:
        plt.imshow(example_image, cmap="gray_r")
        plt.title(f"Example digit (actual label: {y_test[0]})")
        plt.axis("off")
        plt.savefig(IMAGE_PATH, bbox_inches="tight")
        plt.show()
        plt.close()

    return model, accuracy, example_image


if __name__ == "__main__":
    _, accuracy, _ = train_and_save(show_digit=True)
    print(f"Model accuracy: {accuracy:.2%}")
    print(f"Saved model to {MODEL_PATH.name}")
    print(f"Saved test example to {EXAMPLE_PATH.name}")
