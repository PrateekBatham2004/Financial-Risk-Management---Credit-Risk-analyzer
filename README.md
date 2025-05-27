# Credit Risk Manager

This project implements a machine learning model to assess credit default risk based on customer financial information. It is designed for use in financial risk management systems where accurate classification of loan default risk is critical.

## Overview

The model is built using a Random Forest Classifier and includes steps for data preprocessing, model training, evaluation, and model persistence for future use.

## Key Features

- Loads and cleans input credit data
- Selects key features: age, income, loan amount, and loan duration
- Scales data using `StandardScaler` from `scikit-learn`
- Trains a `RandomForestClassifier` to predict credit default risk
- Evaluates the model using:
  - Confusion Matrix
  - F1 Score
  - Precision
  - Recall
- Saves both the trained model and the scaler for future predictions
