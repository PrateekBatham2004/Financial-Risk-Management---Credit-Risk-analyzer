import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score, classification_report
from imblearn.over_sampling import SMOTE # PIP INSTALL IMBALANCED-LEARN
import joblib

# 1. Load the expanded, imbalanced dataset
# Ensure 'credit_data_20_1.csv' is in the same folder
df = pd.read_csv('credit_data_20_1.csv')
df.dropna(inplace=True)

# 2. Preprocessing
# Separate Target
target_col = 'default'
X = df.drop(columns=[target_col])
y = df[target_col]

# Encode Target ('yes'/'no' -> 1/0)
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y) # maps 'no'->0, 'yes'->1

# Encode Categorical Features (One-Hot Encoding)
# This converts text cols like 'checking_balance' into numeric numbers
X = pd.get_dummies(X, drop_first=True)

# 3. Train-Test Split (Stratified to maintain 20:1 ratio in test set)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 4. Apply SMOTE (Synthetic Minority Over-sampling Technique)
# ONLY apply to training data to prevent data leakage
print(f"Original Training Class Distribution: \n{pd.Series(y_train).value_counts()}")
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
print(f"Resampled Training Class Distribution: \n{pd.Series(y_train_resampled).value_counts()}")

# 5. Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_resampled)
X_test_scaled = scaler.transform(X_test)

# Save the scaler
if not os.path.exists('model'):
    os.makedirs('model')
joblib.dump(scaler, 'model/credit_scaler.save')

# 6. Train Random Forest Model
# Using 'class_weight' as an extra safety measure, though SMOTE handles most of it
rf_model = RandomForestClassifier(
    n_estimators=100,
    max_depth=10, # Limited depth to prevent overfitting on synthetic data
    random_state=42,
    n_jobs=-1
)
rf_model.fit(X_train_scaled, y_train_resampled)

# 7. Predictions
y_pred = rf_model.predict(X_test_scaled)
y_prob = rf_model.predict_proba(X_test_scaled)[:, 1] # For ROC-AUC

# 8. Evaluation
print("\n--- Model Evaluation ---")
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print(f"F1 Score: {f1_score(y_test, y_pred):.4f}")
print(f"Precision: {precision_score(y_test, y_pred):.4f}")
print(f"Recall: {recall_score(y_test, y_pred):.4f}")
print(f"ROC-AUC Score: {roc_auc_score(y_test, y_prob):.4f}")

print("\nFull Classification Report:")
print(classification_report(y_test, y_pred, target_names=['No Default', 'Default']))

# 9. Save the trained model
joblib.dump(rf_model, 'model/credit_rf_model.save')
print("Model saved to 'model/credit_rf_model.save'")
