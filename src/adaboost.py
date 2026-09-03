import numpy as np
from stump import DecisionStump


class AdaBoost:
    def __init__(self, T=500):
        self.T = T
        self.stumps = []
        self.stump_weights = []
        self.errors = []
        self.weight_history = []

    def fit(self, X, y):
        m = X.shape[0]
        P = np.ones(m) / m

        for i in range(self.T):
            stump = DecisionStump().fit(X, y, P)
            predictions = stump.predict(X)
            epsilon = np.sum(P[predictions != y])

            if epsilon == 0:
                self.stumps.append(stump)
                self.stump_weights.append(1.0)
                self.errors.append(epsilon)
                self.weight_history.append(P.copy())
                break

            if epsilon >= 0.5:
                break

            w = 0.5 * np.log((1 - epsilon) / epsilon)

            self.stumps.append(stump)
            self.stump_weights.append(w)
            self.errors.append(epsilon)
            self.weight_history.append(P.copy())

            P = P * np.exp(-w * y * predictions)
            P = P / np.sum(P)

        return self

    def predict(self, X):
        total = np.zeros(X.shape[0])
        for stump, w in zip(self.stumps, self.stump_weights):
            total += w * stump.predict(X)
        return np.where(total > 0, 1, -1)

    def staged_error(self, X, y):
        total = np.zeros(X.shape[0])
        errors = []
        for stump, w in zip(self.stumps, self.stump_weights):
            total += w * stump.predict(X)
            pred = np.where(total > 0, 1, -1)
            errors.append(np.mean(pred != y))
        return np.array(errors)