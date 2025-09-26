import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

TARGET_COL = "Churn"

def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df = df.dropna(subset=["TotalCharges"])
    return df

def make_preprocessor(df: pd.DataFrame):
    y = df[TARGET_COL].map({"No": 0, "Yes": 1})
    X = df.drop(columns=[TARGET_COL, "customerID"])

    numeric_cols = X.select_dtypes(include=["int64","float64"]).columns.tolist()
    categorical_cols = X.select_dtypes(include=["object","bool"]).columns.tolist()

    numeric = Pipeline([("imp", SimpleImputer(strategy="median"))])
    categorical = Pipeline([
        ("imp", SimpleImputer(strategy="most_frequent")),
        ("oh", OneHotEncoder(handle_unknown="ignore"))
    ])

    pre = ColumnTransformer([
        ("num", numeric, numeric_cols),
        ("cat", categorical, categorical_cols),
    ])
    return X, y, pre

def split_data(X, y, test_size=0.2, random_state=42):
    return train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)
