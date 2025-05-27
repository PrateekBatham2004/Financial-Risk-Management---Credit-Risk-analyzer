import pandas as pd
import os
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score
import joblib

# Load dataset
df = pd.read_csv('credit_data.csv')
df.dropna(inplace=True)

# Features and target
X = df[['age', 'income', 'amount', 'months_loan_duration']]
y = df['default']  # Target column: 'no' (no default), 'yes' (default)

# Scale the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Save the scaler
if not os.path.exists('model'):
    os.makedirs('model')
joblib.dump(scaler, 'model/credit_scaler.save')

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

# Initialize and train Random Forest model
rf_model = RandomForestClassifier(
    n_estimators=100,
    max_depth=None,
    random_state=42
)
rf_model.fit(X_train, y_train)

# Predictions
y_pred = rf_model.predict(X_test)

# Evaluation - Custom metrics
print("Model trained successfully.")
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred, labels=["no", "yes"]))
print("F1 Score:", f1_score(y_test, y_pred, pos_label="yes"))
print("Precision:", precision_score(y_test, y_pred, pos_label="yes"))
print("Recall:", recall_score(y_test, y_pred, pos_label="yes"))

# Save the trained model
joblib.dump(rf_model, 'model/credit_rf_model.save')
print("Model saved to 'model/credit_rf_model.save'")