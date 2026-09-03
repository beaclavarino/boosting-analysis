import numpy as np
from collections import Counter

from stump import DecisionStump
from datasets import make_xor, make_circle, train_test_split, load_german
from adaboost import AdaBoost
from baselines import run_baselines


def table_sanity():
    print("=== 0. Dataset properties and sanity checks ===")

    for name, gen in [("XOR", make_xor), ("Circle", make_circle)]:
        X, y = gen(seed=0)
        print(f"   {name:8s} positive class: {np.mean(y == 1):.3f}  "
              f"majority baseline error: {1 - max(np.mean(y == 1), np.mean(y == -1)):.3f}")

    X, y, _ = load_german()
    _, ytr, _, yte = train_test_split(X, y, seed=0)
    print(f"   German   positive class: train={np.mean(ytr == 1):.3f} "
          f"test={np.mean(yte == 1):.3f} (overall 0.700)")

    X, y = make_xor(seed=0)
    P = np.ones(len(y)) / len(y)
    s = DecisionStump().fit(X, y, P)
    eps = s._weighted_error(X, y, P, s.feature_index, s.threshold)
    print(f"   single stump on XOR: feature={s.feature_index} "
          f"threshold={s.threshold:.3f} polarity={s.polarity} eps={eps:.3f}")

    tr, te = [], []
    for sd in range(5):
        X, y = make_xor(noise=0.0, seed=sd)
        Xtr, ytr, Xte, yte = train_test_split(X, y, seed=sd)
        m = AdaBoost(T=500).fit(Xtr, ytr)
        tr.append(np.mean(m.predict(Xtr) != ytr))
        te.append(np.mean(m.predict(Xte) != yte))
    print(f"   XOR over 5 seeds: train={np.mean(tr):.3f} test={np.mean(te):.3f} "
          f"gap={np.mean(te) - np.mean(tr):.3f}")

    X, y = make_xor(seed=0)
    Xtr, ytr, Xte, yte = train_test_split(X, y, seed=0)
    m = AdaBoost(T=500).fit(Xtr, ytr)
    pred = m.predict(Xte)
    print(f"   XOR predictions on test: +1={np.sum(pred == 1)} -1={np.sum(pred == -1)}, "
          f"true +1={np.sum(yte == 1)} -1={np.sum(yte == -1)}")
    print()


def table_edges():
    print("=== 1: edge at first and last round (no split) ===")
    for name, gen in [("XOR", lambda: make_xor(noise=0.0, seed=0)),
                      ("Circle", lambda: make_circle(noise=0.0, seed=0))]:
        X, y = gen()
        model = AdaBoost(T=500).fit(X, y)
        e = np.array(model.errors)
        g = 0.5 - e
        err = np.mean(model.predict(X) != y)
        print(f"{name:8s} eps_1={e[0]:.3f} g_1={g[0]:.3f} "
              f"eps_T={e[-1]:.3f} g_T={g[-1]:.3f} train_err={err:.3f}")
    print()


def table_noise(n_seeds=5):
    print(f"=== 2,3,4: noise study (Circle, mean over {n_seeds} seeds) ===")
    for noise in [0.0, 0.1, 0.2]:
        tr_curves, te_curves = [], []
        for s in range(n_seeds):
            X, y = make_circle(noise=noise, seed=s)
            Xtr, ytr, Xte, yte = train_test_split(X, y, seed=s)
            model = AdaBoost(T=500).fit(Xtr, ytr)
            tr_curves.append(model.staged_error(Xtr, ytr))
            te_curves.append(model.staged_error(Xte, yte))
        tr = np.mean(tr_curves, axis=0)
        te = np.mean(te_curves, axis=0)
        print(f"-- noise {noise:.0%}")
        for r in [1, 10, 50, 100, 300, 500]:
            print(f"   T={r:4d}  train={tr[r-1]:.3f}  test={te[r-1]:.3f}")
        print(f"   gap at T=500: {te[-1] - tr[-1]:.3f}")
        print(f"   test minimum: {te.min():.3f} at T={te.argmin() + 1}")
    print()


def table_weights():
    print("=== 4,5: weight distribution (Circle, noise 10%) ===")
    X, y = make_circle(noise=0.1, seed=0)
    _, y_clean = make_circle(noise=0.0, seed=0)
    noisy = (y != y_clean)

    Xtr, ytr, _, _ = train_test_split(X, y, seed=0)
    _, noisy_tr, _, _ = train_test_split(X, noisy, seed=0)
    model = AdaBoost(T=500).fit(Xtr, ytr)

    for r in [1, 10, 50, 100, 300, 500]:
        P = model.weight_history[r - 1]
        print(f"   T={r:4d}  max={P.max():.4f}  top10={np.sort(P)[-10:].sum():.3f}  "
              f"above_uniform={np.sum(P > 1 / len(P))}")

    P = model.weight_history[-1]
    top50 = np.argsort(P)[-50:]
    print(f"   noisy among 50 heaviest: {np.sum(noisy_tr[top50])}/50")
    print(f"   noisy in training set:   {np.sum(noisy_tr)}/{len(noisy_tr)}")
    print()


def table_german_curve():
    print("=== 6. German Credit: error curves ===")
    X, y, _ = load_german()
    Xtr, ytr, Xte, yte = train_test_split(X, y, seed=0)
    model = AdaBoost(T=500).fit(Xtr, ytr)

    tr = model.staged_error(Xtr, ytr)
    te = model.staged_error(Xte, yte)

    for r in [1, 10, 50, 100, 300, 500]:
        print(f"   T={r:4d}  train={tr[r-1]:.3f}  test={te[r-1]:.3f}")
    print(f"   test minimum: {te.min():.3f} at T={te.argmin() + 1}")
    print(f"   gap at T=500: {te[-1] - tr[-1]:.3f}")

    majority = max(np.mean(yte == 1), np.mean(yte == -1))
    print(f"   majority baseline error: {1 - majority:.3f}")
    print()


def table_features():
    print("=== 7: weak learner contribution (German Credit) ===")
    X, y, names = load_german()
    Xtr, ytr, _, _ = train_test_split(X, y, seed=0)
    model = AdaBoost(T=500).fit(Xtr, ytr)

    counts = Counter(s.feature_index for s in model.stumps)
    weights = {}
    for s, w in zip(model.stumps, model.stump_weights):
        weights[s.feature_index] = weights.get(s.feature_index, 0) + w

    for idx, n in counts.most_common(10):
        print(f"   {names[idx]:32s} {n:5d}  {weights[idx]:.3f}")
    print(f"   distinct features used: {len(counts)}/{len(names)}")
    print(f"   w_1={model.stump_weights[0]:.3f}  w_T={model.stump_weights[-1]:.3f}")
    print()


def table_baselines():
    print("=== 8: baselines (test error) ===")
    for name, gen in [("XOR", lambda: make_xor(noise=0.0, seed=0)),
                      ("Circle", lambda: make_circle(noise=0.0, seed=0))]:
        X, y = gen()
        Xtr, ytr, Xte, yte = train_test_split(X, y, seed=0)
        model = AdaBoost(T=500).fit(Xtr, ytr)
        ada = np.mean(model.predict(Xte) != yte)
        base = run_baselines(Xtr, ytr, Xte, yte)
        print(f"{name:8s} ada={ada:.3f} tree={base['tree']:.3f} logreg={base['logreg']:.3f}")

    X, y, _ = load_german()
    Xtr, ytr, Xte, yte = train_test_split(X, y, seed=0)
    model = AdaBoost(T=500).fit(Xtr, ytr)
    ada = np.mean(model.predict(Xte) != yte)
    base = run_baselines(Xtr, ytr, Xte, yte)
    print(f"{'German':8s} ada={ada:.3f} tree={base['tree']:.3f} logreg={base['logreg']:.3f}")
    print()


if __name__ == "__main__":
    table_sanity()
    table_edges()
    table_noise()
    table_weights()
    table_german_curve()
    table_features()
    table_baselines()
    print("tables done")