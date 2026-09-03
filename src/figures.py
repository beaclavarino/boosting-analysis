import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from datasets import make_xor, make_circle, train_test_split, load_german
from adaboost import AdaBoost


def fig_datasets():
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
    for ax, (gen, title) in zip(axes, [(make_xor, "XOR"), (make_circle, "Circle")]):
        X, y = gen(m=500, noise=0.0, seed=0)
        ax.scatter(X[y == 1, 0], X[y == 1, 1], s=8, label="$y = +1$")
        ax.scatter(X[y == -1, 0], X[y == -1, 1], s=8, label="$y = -1$")
        ax.set_title(title)
        ax.set_xlabel("$x_1$")
        ax.set_ylabel("$x_2$")
        ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    fig.savefig("../figures/datasets.png", dpi=150)
    plt.close(fig)


def fig_curves():
    configs = [
        ("XOR", lambda s: make_xor(noise=0.0, seed=s)),
        ("Circle", lambda s: make_circle(noise=0.0, seed=s)),
        ("Circle, 10% label noise", lambda s: make_circle(noise=0.1, seed=s)),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2), sharey=True)

    for ax, (title, gen) in zip(axes, configs):
        X, y = gen(0)
        Xtr, ytr, Xte, yte = train_test_split(X, y, seed=0)
        model = AdaBoost(T=500).fit(Xtr, ytr)
        rounds = np.arange(1, len(model.stumps) + 1)
        ax.plot(rounds, model.staged_error(Xtr, ytr), label="training error")
        ax.plot(rounds, model.staged_error(Xte, yte), label="test error")
        ax.set_title(title)
        ax.set_xlabel("boosting round $T$")
        ax.set_xscale("log")
        ax.legend(fontsize=8)

    axes[0].set_ylabel("zero-one loss")
    fig.tight_layout()
    fig.savefig("../figures/curves.png", dpi=150)
    plt.close(fig)


def fig_overfitting(n_seeds=5):
    noises = [0.0, 0.1, 0.2]
    fig, ax = plt.subplots(figsize=(7, 4.5))

    for noise in noises:
        curves = []
        for s in range(n_seeds):
            X, y = make_circle(noise=noise, seed=s)
            Xtr, ytr, Xte, yte = train_test_split(X, y, seed=s)
            model = AdaBoost(T=500).fit(Xtr, ytr)
            curves.append(model.staged_error(Xte, yte))
        mean_curve = np.mean(curves, axis=0)
        rounds = np.arange(1, len(mean_curve) + 1)
        ax.plot(rounds, mean_curve, label=f"noise = {noise:.0%}")

        best = np.argmin(mean_curve)
        ax.plot(best + 1, mean_curve[best], "o", ms=5,
                color=ax.lines[-1].get_color())

    ax.set_xscale("log")
    ax.set_xlabel("boosting round $T$")
    ax.set_ylabel("test error")
    ax.set_title("Test error vs. rounds, averaged over 5 seeds (Circle)")
    ax.legend()
    fig.tight_layout()
    fig.savefig("../figures/overfitting.png", dpi=150)
    plt.close(fig)


def fig_gammas():
    configs = [
        ("XOR", lambda s: make_xor(noise=0.0, seed=s)),
        ("Circle", lambda s: make_circle(noise=0.0, seed=s)),
    ]
    fig, ax = plt.subplots(figsize=(7, 4.5))

    for title, gen in configs:
        X, y = gen(0)
        Xtr, ytr, _, _ = train_test_split(X, y, seed=0)
        model = AdaBoost(T=500).fit(Xtr, ytr)
        gammas = 0.5 - np.array(model.errors)
        ax.plot(np.arange(1, len(gammas) + 1), gammas, label=title)

    ax.set_xscale("log")
    ax.set_xlabel("boosting round $i$")
    ax.set_ylabel(r"edge $\gamma_i = 1/2 - \varepsilon_i$")
    ax.set_title("Weak learner edge over rounds")
    ax.legend()
    fig.tight_layout()
    fig.savefig("../figures/gammas.png", dpi=150)
    plt.close(fig)


def fig_german():
    X, y, _ = load_german()
    Xtr, ytr, Xte, yte = train_test_split(X, y, seed=0)
    model = AdaBoost(T=500).fit(Xtr, ytr)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    rounds = np.arange(1, len(model.stumps) + 1)
    test_curve = model.staged_error(Xte, yte)
    ax.plot(rounds, model.staged_error(Xtr, ytr), label="training error")
    ax.plot(rounds, test_curve, label="test error")
    majority = max(np.mean(yte == 1), np.mean(yte == -1))
    ax.axhline(1 - majority, ls="--", c="gray", lw=1, label="majority baseline")
    
    best = np.argmin(test_curve)
    ax.plot(best + 1, test_curve[best], "o", ms=5, c="tab:orange")

    ax.set_xscale("log")
    ax.set_xlabel("boosting round $T$")
    ax.set_ylabel("zero-one loss")
    ax.set_title("German Credit")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig("../figures/german.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    fig_datasets()
    fig_curves()
    fig_overfitting()
    fig_gammas()
    fig_german()
    print("figures done")