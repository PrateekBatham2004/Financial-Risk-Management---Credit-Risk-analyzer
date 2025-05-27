# Credit Risk Manager

This project is a financial risk management tool using a Random Forest classifier to predict credit default risks.

## Features

- Preprocesses and scales credit data
- Trains a Random Forest model on key financial features
- Evaluates performance using F1 Score, Precision, Recall, and Confusion Matrix
- Saves the scaler and trained model for reuse

## Files

- `main.py`: Main training and evaluation script
- `model/`: Contains the saved scaler and model files after training
- `credit_data.csv`: Input dataset (not included, user must provide)

## Usage

1. Place your `credit_data.csv` file in the project root directory.
2. Run the script:
    ```bash
    python main.py
    ```