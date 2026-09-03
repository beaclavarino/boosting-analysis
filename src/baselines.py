import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline


def run_baselines(Xtr, ytr, Xte, yte, seed=0):
    results = {}

    tree = DecisionTreeClassifier(random_state=seed)
    tree.fit(Xtr, ytr)
    results["tree"] = np.mean(tree.predict(Xte) != yte)

    logreg = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=5000, random_state=seed),
    )
    logreg.fit(Xtr, ytr)
    results["logreg"] = np.mean(logreg.predict(Xte) != yte)

    return results