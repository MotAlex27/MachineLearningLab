# Machine Learning Classifiers

This repository contains two independent, dependency-free Python classifiers.
Each algorithm has its own source code, tests, documentation, and datasets.

## Algorithms

| Algorithm | Source | Tests | Documentation |
|---|---|---|---|
| ID3 decision tree | [decision_tree/src/](./decision_tree/src/) | [decision_tree/tests/](./decision_tree/tests/) | [Decision Tree README](./decision_tree/README.md) |
| Bernoulli Naïve Bayes | [naive_bayes/src/](./naive_bayes/src/) | [naive_bayes/tests/](./naive_bayes/tests/) | [Naïve Bayes README](./naive_bayes/README.md) |

Datasets are kept with the algorithm that uses them. Generated results are
written to the root `results/` folder.

Run all tests from the repository root:

```powershell
python -m unittest
```
