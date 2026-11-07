"""Data loading, feature building and time-based splitting."""
import numpy as np
import pandas as pd

TARGET = "Class"


def load_raw(path="data/creditcard.csv"):
    """Load the ULB credit card CSV and sort it by time (oldest first)."""
    df = pd.read_csv(path)
    missing = {"Time", "Amount", TARGET} - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {missing}")
    return df.sort_values("Time").reset_index(drop=True)


def add_features(df):
    """Add two simple features. V1..V28 are already anonymised PCA components."""
    out = df.copy()
    out["hour"] = ((out["Time"] // 3600) % 24).astype(int)  # hours since start, wrapped to 24
    out["log_amount"] = np.log1p(out["Amount"])
    return out


def feature_columns(df):
    """Model inputs. 'Time' and the label are deliberately excluded."""
    v_cols = [c for c in df.columns if c.startswith("V")]
    return v_cols + ["log_amount", "hour"]


def time_split(df, train_frac=0.6, val_frac=0.2):
    """Split by time: train = oldest rows, validation = next, test = newest."""
    df = df.sort_values("Time").reset_index(drop=True)
    n = len(df)
    i, j = int(n * train_frac), int(n * (train_frac + val_frac))
    return df.iloc[:i], df.iloc[i:j], df.iloc[j:]