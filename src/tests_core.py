import numpy as np

from stump import DecisionStump
from adaboost import AdaBoost
from datasets import make_xor, make_circle, train_test_split, load_german


def test_stump_separable():
    X = np.array([[1.0], [2.0], [3.0], [4.0]])
    y = np.array([-1, -1, 1, 1])
    w = np.ones(4) / 4
    s = DecisionStump().fit(X, y, w)
    assert s.threshold == 2.5
    assert s.polarity == 1
    assert np.mean(s.predict(X) != y) == 0.0


def test_stump_inverted_labels():
    X = np.array([[1.0], [2.0], [3.0], [4.0]])
    y = np.array([1, 1, -1, -1])
    w = np.ones(4) / 4
    s = DecisionStump().fit(X, y, w)
    assert s.threshold == 2.5
    assert s.polarity == -1
    assert np.mean(s.predict(X) != y) == 0.0


def test_candidate_thresholds():
    s = DecisionStump()
    column = np.array([30, 25, 60, 45, 30])
    result = s._candidate_thresholds(column)
    assert np.allclose(result, [27.5, 37.5, 52.5])


def test_stump_xor_is_weak():
    X, y = make_xor(seed=0)
    w = np.ones(len(y)) / len(y)
    s = DecisionStump().fit(X, y, w)
    err = s._weighted_error(X, y, w, s.feature_index, s.threshold)
    assert 0.40 < err < 0.50


def test_adaboost_separable():
    X = np.array([[1.0], [2.0], [3.0], [4.0]])
    y = np.array([-1, -1, 1, 1])
    model = AdaBoost(T=10).fit(X, y)
    assert len(model.stumps) == 1
    assert np.mean(model.predict(X) != y) == 0.0


def test_distributions_sum_to_one():
    X, y = make_circle(noise=0.1, seed=0)
    model = AdaBoost(T=50).fit(X, y)
    for P in model.weight_history:
        assert np.isclose(P.sum(), 1.0)


def test_epsilon_never_above_half():
    X, y = make_circle(noise=0.2, seed=0)
    model = AdaBoost(T=100).fit(X, y)
    for eps in model.errors:
        assert eps <= 0.5


def test_history_lengths_match():
    X, y = make_circle(noise=0.1, seed=0)
    model = AdaBoost(T=50).fit(X, y)
    n = len(model.stumps)
    assert len(model.stump_weights) == n
    assert len(model.errors) == n
    assert len(model.weight_history) == n


def test_weight_history_is_copied():
    X, y = make_circle(noise=0.1, seed=0)
    model = AdaBoost(T=20).fit(X, y)
    assert not np.allclose(model.weight_history[0], model.weight_history[-1])


def test_training_error_decreases():
    X, y = make_circle(noise=0.0, seed=0)
    model = AdaBoost(T=200).fit(X, y)
    curve = model.staged_error(X, y)
    assert curve[-1] < curve[0]


def test_matches_sklearn():
    from sklearn.ensemble import AdaBoostClassifier
    from sklearn.tree import DecisionTreeClassifier

    X, y = make_circle(noise=0.1, seed=0)
    mine = AdaBoost(T=50).fit(X, y)
    err_mine = np.mean(mine.predict(X) != y)

    ref = AdaBoostClassifier(
        estimator=DecisionTreeClassifier(max_depth=1),
        n_estimators=50,
        random_state=0,
    )
    ref.fit(X, y)
    err_ref = np.mean(ref.predict(X) != y)

    assert abs(err_mine - err_ref) < 0.05


if __name__ == "__main__":
    test_stump_separable()
    test_stump_inverted_labels()
    test_candidate_thresholds()
    test_stump_xor_is_weak()
    test_adaboost_separable()
    test_distributions_sum_to_one()
    test_epsilon_never_above_half()
    test_history_lengths_match()
    test_weight_history_is_copied()
    test_training_error_decreases()
    test_matches_sklearn()
    print("all tests passed")