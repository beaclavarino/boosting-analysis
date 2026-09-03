# Boosting and Additive Models

**Statistical Methods for Machine Learning** — Master in Data Science for Economics

*Beatrice Clavarino*

The task is to implement and analyse AdaBoost (Freund & Schapire, 1997), using decision stumps as weak learners. The algorithm is evaluated on three binary classification problems — two synthetic (XOR, Circle) and one real-world (German Credit, UCI Machine Learning Repository) — studying the evolution of training/test error, the weak-learner edge, overfitting under label noise, and a comparison against a decision tree and logistic regression.

Repository structure:

* `report.pdf`: The project report, written in LaTeX, provides a comprehensive overview of the methodology, experiments, and results.
* `src/`: Python source code used for the project (`datasets.py`, `stump.py`, `adaboost.py`, `baselines.py`, `figures.py`, `tables.py`).
* `data/`: German Credit dataset (UCI).
* `figures/`: Plots included in the report.
