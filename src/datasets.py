import numpy as np
import pandas as pd


def make_xor(m=1000, noise=0.0, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.uniform(-1, 1, size=(m, 2))
    y = np.where(X[:, 0] * X[:, 1] > 0, 1, -1)
    if noise > 0:
        flip = rng.random(m) < noise
        y[flip] = -y[flip]
    return X, y

def make_circle(m=1000, noise=0.0, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.uniform(-1, 1, size=(m, 2))
    radius = np.sqrt(X[:, 0] ** 2 + X[:, 1] ** 2)
    y = np.where(radius > 0.6, 1, -1)
    if noise > 0:
        flip = rng.random(m) < noise
        y[flip] = -y[flip]
    return X, y

def train_test_split(X, y, test_frac=0.3, seed=0):
    rng = np.random.default_rng(seed)
    m = len(y)
    idx = rng.permutation(m)
    n_test = int(m * test_frac)
    test_idx = idx[:n_test]
    train_idx = idx[n_test:]
    return X[train_idx], y[train_idx], X[test_idx], y[test_idx]


def load_german(path="../data/german.data"):
    columns = [
        "checking_account", "duration", "credit_history", "purpose",
        "credit_amount", "savings", "employment_since", "installment_rate",
        "personal_status", "other_debtors", "residence_since", "property",
        "age", "other_installment_plans", "housing", "existing_credits",
        "job", "liable_people", "telephone", "foreign_worker", "target",
    ]
    df = pd.read_csv(path, sep=" ", header=None, names=columns)
    y = np.where(df["target"] == 1, 1, -1)
    X_df = df.drop(columns=["target"])
    X_df = pd.get_dummies(X_df)
    return X_df.values.astype(float), y, list(X_df.columns)