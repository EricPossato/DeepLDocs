import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

# Load dataset
df = pd.read_csv("./docs/exercise1/titanic_dataset/train.csv")

# Handle Missing Data

# Numerical: fill with median
num_cols = ["Age", "RoomService", "FoodCourt", "ShoppingMall", "Spa", "VRDeck"]
for col in num_cols:
    df[col] = df[col].fillna(df[col].median())

# Categorical: fill with mode
cat_cols = ["HomePlanet", "CryoSleep", "Destination", "VIP"]
for col in cat_cols:
    # make column string dtype
    df[col] = df[col].astype("string")
    # fill NaN with mode and convert to string dtype
    df[col] = df[col].fillna(df[col].mode(dropna=True)[0]).astype(str)

# Drop Cabin and Name
df.drop(["Cabin", "Name"], axis=1, inplace=True)

# Convert CryoSleep and VIP from True/False strings to 0/1 ints
df["CryoSleep"] = df["CryoSleep"].map({"True": 1, "False": 0})
df["VIP"] = df["VIP"].map({"True": 1, "False": 0})

# Encode Categorical
df["CryoSleep"] = df["CryoSleep"].astype(int)
df["VIP"] = df["VIP"].astype(int)

# One-hot encode categorical vars
df = pd.get_dummies(df, columns=["HomePlanet", "Destination"], drop_first=True)

# Capture copy BEFORE scaling
pre_scale_num = df[num_cols].copy()

# Scale Numerical Features
scaler = MinMaxScaler(feature_range=(-1, 1))
df[num_cols] = scaler.fit_transform(df[num_cols].astype("float64"))

print(df.head())

# histograms before vs after
features_to_show = ["Age", "FoodCourt"]  # pick any subset of num_cols

for col in features_to_show:
    if col in pre_scale_num.columns:
        fig, ax = plt.subplots(1, 2, figsize=(10, 4), sharex=False, sharey=False)
        ax[0].hist(pre_scale_num[col].dropna(), bins=30)
        ax[0].set_title(f"{col} — before scaling")
        ax[0].set_xlabel(col); ax[0].set_ylabel("count")

        ax[1].hist(df[col].dropna(), bins=30)
        ax[1].set_title(f"{col} — after scaling (standardized)")
        ax[1].set_xlabel(col); ax[1].set_ylabel("count")

        plt.tight_layout()
        # Save figures
        plt.savefig(f"hist_{col.lower()}_before_after.png", dpi=150, bbox_inches="tight")
        plt.show()
