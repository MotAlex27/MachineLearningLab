# Decision Tree Classifier

This project is a dependency-free Python application that reads an ARFF
dataset, splits it into training and testing files, trains an ID3 decision
tree, and evaluates the model using classification accuracy.

## What the program does

The pipeline in [decision_tree.py](./decision_tree.py) is:

1. Read the ARFF relation, attribute declarations, and data rows.
2. Select the target/class attribute (the last attribute by default).
3. Randomly divide the rows into training and testing sets. The default is
   approximately 70% training and 30% testing.
4. Convert declared numeric input attributes into three discrete, equal-width
   bins. Bin boundaries are learned from training rows, then reused for test
   rows so information from the test set does not affect training.
5. Train an ID3 decision tree using the training rows.
6. Predict the class of each testing row and compute accuracy.
7. Save both processed datasets, a readable tree, and evaluation details.

The two provided datasets declare their inputs as nominal. In particular,
Soybean's values that look like numbers are category codes, not measurements,
so they remain categories and are not incorrectly binned.

## Decision-tree theory

A decision tree predicts a class by asking a sequence of questions about
attributes. Each internal node tests one attribute, each outgoing branch
corresponds to one of that attribute's values, and each leaf predicts a class.
For example:

```text
outlook = overcast:
  class: yes
outlook = sunny:
  humidity = high:
    class: no
```

To choose the next question, ID3 uses **entropy**, a measure of how mixed the
classes are. If there are `k` classes and class `i` occurs with proportion
`p_i`, entropy is:

```text
H(S) = -sum(p_i * log2(p_i))
```

Entropy is zero when all examples have the same class and is higher when
classes are mixed. For a candidate attribute `A`, the examples are grouped by
its values. The remaining entropy after splitting is the weighted sum of the
groups' entropies:

```text
H(S | A) = sum over values v of A:
           (number of examples with A=v / number of examples) * H(S_v)
```

The **information gain** is:

```text
Gain(S, A) = H(S) - H(S | A)
```

The program chooses the available attribute with the greatest positive gain
and repeats the process recursively for each resulting group. A node becomes
a leaf when all its examples have the same class, no attributes remain, or no
attribute gives positive information gain. The leaf predicts the most common
class in its group. The same attribute is not tested again lower down that
path. For a test row containing a branch value not observed during training,
prediction falls back to the current node's majority class.

This is the categorical ID3 algorithm. Numeric measurements need to become
categories first because these splits test values as discrete branches rather
than searching for continuous thresholds.

## Numeric preprocessing and missing values

Only attributes declared `numeric`, `real`, or `integer` in the ARFF header
are treated as numeric. For each such input attribute, the program finds the
minimum and maximum among the training examples and divides that range into
three equal-width intervals:

```text
width = (maximum - minimum) / 3
bin_1: x <= minimum + width
bin_2: minimum + width < x <= minimum + 2 * width
bin_3: x > minimum + 2 * width
```

The learned thresholds are applied unchanged to the testing rows. A numeric
attribute with no observed training values is left with three possible bin
labels, but no thresholds to apply. Missing values written as `?` are
preserved and treated as a category by the tree; the application does not
impute missing data. The class attribute is never binned.

An ARFF declaration such as `@attribute code {0, 1, 2}` is nominal even though
its values look numeric. An actual measurement is declared using, for
example, `@attribute temperature numeric`.

## Train/test split and evaluation

The split is stratified: rows are grouped by their class, shuffled, and each
class group is divided using the requested training fraction. This helps keep
class proportions similar across both sets. A random seed makes the split
repeatable. Small or rare classes may make the final overall percentage
slightly different from exactly 70/30; the program keeps at least one example
of each class in training when possible.

Accuracy is the share of test examples whose predicted class matches the true
class:

```text
accuracy = correctly classified test rows / total test rows
```

Accuracy is easy to interpret but can be misleading for imbalanced datasets:
a model can score well by favoring a common class. For a fuller evaluation,
also consider per-class precision/recall and a confusion matrix.

## Run the application

Python 3.9 or newer is recommended. Only the Python standard library is used;
no package installation is required. Run commands from this project directory.
With no dataset argument, the application uses `soybean_modify.arff`:

```powershell
python .\decision_tree.py
```

Choose either supplied dataset explicitly:

```powershell
python .\decision_tree.py .\weather.nominal.arff --seed 42
python .\decision_tree.py .\soybean_modify.arff --seed 42
```

Options:

| Option | Default | Description |
|---|---|---|
| `dataset` | `soybean_modify.arff` | Input ARFF path |
| `--train-fraction` | `0.7` | Fraction of each class group assigned to training; must be between 0 and 1 |
| `--seed` | `42` | Random seed used to shuffle rows |
| `--class-attribute` | Last attribute | Name of the target attribute |
| `--output-dir` | `results\<dataset-name>\` | Directory in which to write the results |

For example, to classify Weather with an 80/20 split and select `play` as the
target:

```powershell
python .\decision_tree.py .\weather.nominal.arff --train-fraction 0.8 --class-attribute play
```

Paths are interpreted relative to the current working directory. Input files
are read but never modified.

## Output files

By default, the output directory is `results\<input-file-name-without-extension>\`.
For example, running on `weather.nominal.arff` creates:

- `training_set.arff` — the shuffled, preprocessed training rows.
- `testing_set.arff` — the shuffled, preprocessed testing rows.
- `decision_tree.txt` — the learned tree in an indented, human-readable form.
- `evaluation.txt` — row counts, number correctly classified, accuracy, and
  random seed.

The same counts and accuracy are also printed in the terminal. The result
directory is created automatically. Use `--output-dir` to specify another
location.

## How the code is organized

The implementation is intentionally small and uses plain Python data
structures:

- `Attribute`, `Dataset`, and `Tree` are dataclasses describing the ARFF
  schema, rows, and recursive tree nodes.
- `parse_arff` reads the relation, nominal/numeric declarations, and records;
  it validates row widths and reports malformed or unsupported declarations.
- `split_rows` groups and shuffles rows by class to create a reproducible
  stratified split.
- `fit_numeric_bins`, `discretize`, and `transform_rows` learn bin thresholds
  from training data and apply the same mapping to train and test data.
- `entropy`, `majority_label`, and `build_tree` implement ID3.
- `predict` follows tree branches to a leaf, using a node's majority class if
  a branch is unknown.
- `write_arff` serializes the processed splits; `format_tree` prints the tree.
- `run` coordinates loading, splitting, training, testing, and output.
- `main` parses command-line options and reports user-facing input errors.

The tests in [test_decision_tree.py](./test_decision_tree.py) check numeric
binning, tree learning and fallback prediction, deterministic splitting, and
the distinction between nominal numeric-looking codes and numeric attributes.
Run them with:

```powershell
python -m unittest .\test_decision_tree.py
```

## Scope and limitations

- The parser supports ARFF nominal attributes and numeric/real/integer
  attributes, comments beginning with `%`, and ordinary comma-separated data
  rows. It is not a complete implementation of every ARFF feature.
- ID3 uses information gain, which can favor attributes with many distinct
  values. This implementation does not prune the learned tree, smooth
  probabilities, or calculate confidence scores.
- Each nominal value creates a separate branch. Numeric discretization uses
  simple equal-width bins, not bins optimized for class separation.
- Missing values are treated as their own category, not estimated.
- Stratification and rounding are approximate for very small class groups.
- Accuracy is measured only on the held-out test set; results depend on the
  selected split and seed.