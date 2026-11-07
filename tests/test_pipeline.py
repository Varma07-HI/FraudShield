import numpy as np
import pandas as pd

from src.data import add_features, feature_columns, time_split
from src.metrics import threshold_table


def make_df(n=1000):
    rng = np.random.default_rng(0)
    df = pd.DataFrame(rng.normal(size=(n, 28)), columns=[f"V{i}" for i in range(1, 29)])
    df["Time"] = rng.permutation(n).astype(float) * 100
    df["Amount"] = rng.uniform(1, 200, n)
    df["Class"] = (rng.random(n) < 0.05).astype(int)
    return df


def test_split_is_chronological_and_disjoint():
    train, val, test = time_split(add_features(make_df()))
    assert train["Time"].max() < val["Time"].min()
    assert val["Time"].max() < test["Time"].min()
    assert len(train) + len(val) + len(test) == 1000


def test_features_exclude_time_and_label():
    feats = feature_columns(add_features(make_df()))
    assert "Time" not in feats and "Class" not in feats
    assert "log_amount" in feats and "hour" in feats


def test_threshold_table_flags_fewer_as_threshold_rises():
    rng = np.random.default_rng(1)
    t = threshold_table(rng.integers(0, 2, 500), rng.random(500), rng.uniform(1, 100, 500))
    assert t["flagged"].is_monotonic_decreasing