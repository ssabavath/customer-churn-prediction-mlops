import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    roc_auc_score
)

# Load data
df = pd.read_csv("data/processed_churn.csv")

# Features and target
X = df.drop("Churn", axis=1)
y = df["Churn"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Load trained model
model = joblib.load("models/churn_model.pkl")

# Predictions
pred = model.predict(X_test)

# Evaluation
print("\nConfusion Matrix")
print(confusion_matrix(y_test, pred))

print("\nClassification Report")
print(classification_report(y_test, pred))

# ROC AUC
prob = model.predict_proba(X_test)[:, 1]

print("\nROC-AUC Score:")
print(roc_auc_score(y_test, prob))