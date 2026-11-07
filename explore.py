import pandas as pd

df = pd.read_csv("data/creditcard.csv")
print(df.shape)
print(df["Class"].value_counts())
print(f"Fraud rate: {df['Class'].mean():.4%}")
print(df[["Time", "Amount"]].describe())
print(df.groupby("Class")["Amount"].describe())
print(df.isna().sum().sum(), "missing values")