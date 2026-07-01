import pandas as pd

df = pd.read_csv("data/WA_Fn-UseC_-Telco-Customer-Churn.csv")

print("="*50)
print("FIRST 5 ROWS")
print(df.head())

print("="*50)
print("SHAPE")
print(df.shape)

print("="*50)
print("INFO")
print(df.info())

print("="*50)
print("MISSING VALUES")
print(df.isnull().sum())

print("="*50)
print("CHURN COUNTS")
print(df["Churn"].value_counts())