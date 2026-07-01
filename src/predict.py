import pandas as pd
import joblib

model = joblib.load(
    "models/churn_model.pkl"
)

sample = pd.read_csv(
    "data/processed_churn.csv"
)

sample = sample.drop(
    "Churn",
    axis=1
)

customer = sample.iloc[[0]]

prediction = model.predict(customer)

if prediction[0] == 1:
    print("Customer Will Churn")
else:
    print("Customer Will Stay")