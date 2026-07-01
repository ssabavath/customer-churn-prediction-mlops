from fastapi import FastAPI
import joblib
import pandas as pd

app = FastAPI()

# Load model
model = joblib.load("models/churn_model.pkl")

@app.get("/")
def home():
    return {"message": "Customer Churn Prediction API"}

@app.get("/predict")
def predict():

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
        result = "Customer Will Churn"
    else:
        result = "Customer Will Stay"

    return {"prediction": result}