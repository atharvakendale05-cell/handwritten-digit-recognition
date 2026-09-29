"""Small tests that check the training and prediction workflow."""

import numpy as np

from train_model import train_and_save


def test_model_is_reasonably_accurate():
    _, accuracy, _ = train_and_save()
    assert accuracy >= 0.90


def test_model_predicts_a_valid_digit():
    model, _, example_image = train_and_save()
    prediction = model.predict(example_image.reshape(1, -1))[0]
    assert prediction in range(10)
    assert np.asarray(example_image).shape == (8, 8)
