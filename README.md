# Handwritten Digit Recognition

This beginner-friendly project teaches the basic machine-learning workflow by recognizing handwritten digits from 0 to 9.

It uses the small `digits` dataset included with scikit-learn. Each example is an 8x8 grayscale image represented as 64 numbers. A `KNeighborsClassifier` looks at nearby training examples and chooses the most common digit label.

## Setup

Create and activate a virtual environment (recommended), then install the dependencies:

```bash
python -m venv .venv
# Windows PowerShell
.venv\Scripts\Activate.ps1
# macOS/Linux
source .venv/bin/activate
pip install -r requirements.txt
```

## Train the model

```bash
python train_model.py
```

This command splits the data into training and testing sets, trains the model, prints its accuracy, displays a test digit with matplotlib, and saves:

- `digit_model.joblib` — the trained model.
- `test_example.npy` — one saved 8x8 test image.
- `digit_example.png` — a copy of the displayed digit.

Close the matplotlib window to let the script finish.

## Make a prediction

After training, run:

```bash
python predict.py
```

The script loads the saved model and saved test image, then prints the predicted digit.

## Run the tests

```bash
pytest -q
```

## Files

- `train_model.py`: loads the dataset, splits it, trains KNN, evaluates accuracy, displays an example, and saves artifacts.
- `predict.py`: loads the saved artifacts and predicts the saved example.
- `test_model.py`: checks that accuracy is good enough and that predictions are valid digits.
- `requirements.txt`: lists the Python packages needed by the project.
- `.gitignore`: keeps generated files, caches, and virtual environments out of Git.

No OpenAI API key or external API is needed.
