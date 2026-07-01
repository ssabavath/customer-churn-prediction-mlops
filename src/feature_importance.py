import pandas as pd
import joblib
import matplotlib.pyplot as plt

model = joblib.load(
    "models/churn_model.pkl"
)

df = pd.read_csv(
    "data/processed_churn.csv"
)

X = df.drop("Churn", axis=1)

importance = model.feature_importances_

feature_importance = pd.DataFrame({
    "Feature": X.columns,
    "Importance": importance
})

feature_importance = feature_importance.sort_values(
    by="Importance",
    ascending=False
)

print(feature_importance.head(10))

plt.figure(figsize=(10,6))
plt.barh(
    feature_importance["Feature"][:10],
    feature_importance["Importance"][:10]
)
plt.show()