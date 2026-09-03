import numpy as np


class DecisionStump:
    def __init__(self):
        self.feature_index = None
        self.threshold = None
        self.polarity = 1

    def predict(self, X):
        feature_values = X[:, self.feature_index]
        return self.polarity * np.where(feature_values > self.threshold, 1, -1)

    def _weighted_error(self, X, y, sample_weight, feature_index, threshold):
        predictions = np.where(X[:, feature_index] > threshold, 1, -1)
        return np.sum(sample_weight[predictions != y])    

    def _candidate_thresholds(self, column):
        values = np.unique(column)
        return (values[:-1] + values[1:]) / 2

    def fit(self, X, y, sample_weight):
        m, d = X.shape
        best_error = np.inf

        for feature_index in range(d):
            thresholds = self._candidate_thresholds(X[:, feature_index])
            for threshold in thresholds:
                error = self._weighted_error(X, y, sample_weight, feature_index, threshold)
                polarity = 1
                if error > 0.5:
                    error = 1 - error
                    polarity = -1
                if error < best_error:
                    best_error = error
                    self.feature_index = feature_index
                    self.threshold = threshold
                    self.polarity = polarity
        return self