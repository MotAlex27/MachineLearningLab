# Reuters Naïve Bayes Classifier

This dependency-free application implements Bernoulli Naïve Bayes for the
preprocessed Reuters sparse document vectors. Source is in
[src/naive_bayes.py](./src/naive_bayes.py), tests are in
[tests/test_naive_bayes.py](./tests/test_naive_bayes.py), and the input files
are in [data/Reuters_Dataset/](./data/Reuters_Dataset/).

## Input and algorithm

Each Reuters document is a sparse vector. The program interprets each
attribute index as its position in the ordered `@attribute` declarations and
converts every positive term frequency to a binary present/absent feature.
When a document has multiple topics after `#`, only its first topic is used.
The supplied training and testing files are evaluated as-is; no additional
random split is made.

For each class, the classifier estimates word-presence probabilities with
Laplace smoothing (alpha 1 by default). Predictions use the class prior and
the Bernoulli probabilities of both present and absent features.

## Run

From the repository root, run with the default 1309-attribute training and
testing files:

```powershell
python .\naive_bayes\src\naive_bayes.py
```

To use the 100-attribute pair:

```powershell
python .\naive_bayes\src\naive_bayes.py `
  .\naive_bayes\data\Reuters_Dataset\MultiClass_IG_Training_Chose_100.0.arff `
  .\naive_bayes\data\Reuters_Dataset\MultiClass_IG_Testing_Chose_100.0.arff
```

Use `--alpha` to change the positive smoothing value and `--output-dir` to
choose another results directory. By default, evaluation output is written
under `results/` at the repository root. `evaluation.txt` reports accuracy
and per-class precision, recall, and F1; `predictions.csv` contains the
actual and predicted topic for every test document.

Run the Naïve Bayes tests from the repository root:

```powershell
python -m unittest naive_bayes.tests.test_naive_bayes
```

## Test results

On the bundled files with the default smoothing value, the 1309-attribute
pair achieved 66.65% accuracy and the 100-attribute pair achieved 64.06%.
