import pandas as pd

# Load dataset (adjust path to your train.csv)
df = pd.read_csv("./docs/exercise1/titanic_dataset/train.csv")


# List all columns
print("\nColumns in dataset:")
print(df.columns.tolist())

# Show data types and non-null counts
print("\nInfo about dataset:")
print(df.info())


# Quick separation into numerical vs categorical
numerical_cols = df.select_dtypes(include=["int64", "float64"]).columns.tolist()
categorical_cols = df.select_dtypes(include=["object", "bool"]).columns.tolist()

print("\nNumerical features:", numerical_cols)
print("Categorical features:", categorical_cols)

# Summary of missing values
print("\nMissing values per column:")
print(df.isna().sum())