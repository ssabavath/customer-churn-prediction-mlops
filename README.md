# Customer Churn Prediction - End-to-End ML Project

## Project Overview

This project predicts whether a customer is likely to churn (leave a service) based on customer demographics, account information, and service usage data.

The project follows a complete Machine Learning workflow including:

- Data Analysis
- Data Preprocessing
- Feature Engineering
- Model Training
- Model Evaluation
- Hyperparameter Tuning
- Model Deployment using FastAPI
- Containerization using Docker

---

## Business Problem

Customer churn is one of the biggest challenges for subscription-based businesses.

Retaining existing customers is significantly cheaper than acquiring new customers.

This project helps identify customers who are likely to leave so businesses can take proactive actions.

---

## Dataset

Dataset: Telco Customer Churn Dataset

Features include:

- Gender
- Senior Citizen
- Partner
- Dependents
- Tenure
- Internet Service
- Contract Type
- Monthly Charges
- Total Charges

Target Variable:

```text
Churn
```

- Yes = Customer leaves
- No = Customer stays

---

## Project Structure

```text
customer-churn-prediction/
│
├── api/
│   └── main.py
│
├── data/
│
├── models/
│   └── churn_model.pkl
│
├── src/
│   ├── eda.py
│   ├── preprocess.py
│   ├── train.py
│   ├── evaluate.py
│   ├── predict.py
│   ├── feature_importance.py
│   └── tune_model.py
│
├── Dockerfile
├── requirements.txt
└── README.md
```

---

## Technologies Used

### Programming

- Python

### Data Analysis

- Pandas
- NumPy

### Visualization

- Matplotlib
- Seaborn

### Machine Learning

- Scikit-Learn
- Random Forest
- XGBoost

### Deployment

- FastAPI
- Docker

---

## Machine Learning Workflow

```text
Raw Data
    ↓
EDA
    ↓
Data Cleaning
    ↓
Encoding
    ↓
Feature Engineering
    ↓
Train/Test Split
    ↓
Random Forest
    ↓
Hyperparameter Tuning
    ↓
Prediction
    ↓
FastAPI
    ↓
Docker
```

---

## Exploratory Data Analysis

Performed:

- Missing Value Analysis
- Data Type Inspection
- Churn Distribution Analysis
- Feature Exploration

---

## Data Preprocessing

Steps:

1. Removed customerID
2. Converted TotalCharges to numeric
3. Handled missing values
4. Encoded categorical variables
5. Generated processed dataset

---

## Model Training

Model Used:

### Random Forest Classifier

Parameters tuned using:

- GridSearchCV

Best Parameters:

```python
{
    'max_depth': 10,
    'min_samples_split': 2,
    'n_estimators': 200
}
```

---

## Feature Importance

Top Important Features:

- TotalCharges
- MonthlyCharges
- Tenure
- Contract
- PaymentMethod

These features have the strongest influence on customer churn.

---

## Model Evaluation

Metrics Used:

- Accuracy
- Precision
- Recall
- F1 Score
- ROC-AUC

---

## FastAPI Deployment

### Start Server

```bash
uvicorn api.main:app --reload
```

### Home Endpoint

```text
http://127.0.0.1:8000
```

### Prediction Endpoint

```text
http://127.0.0.1:8000/predict
```

---

## Docker Deployment

### Build Docker Image

```bash
docker build -t churn-api .
```

### Run Docker Container

```bash
docker run -p 8000:8000 churn-api
```

### Open API

```text
http://localhost:8000
```

---

## Sample Output

```json
{
  "prediction": "Customer Will Stay"
}
```

---

## Future Improvements

- XGBoost Optimization
- MLflow Integration
- Azure Deployment
- CI/CD with GitHub Actions
- Real User Input API
- Streamlit Dashboard

---

