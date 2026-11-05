#!/usr/bin/env python3
"""Train and evaluate an ID3 decision tree from an ARFF dataset."""

from __future__ import annotations

import argparse
import csv
import math
import random
import re
import sys
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

PROJECT_ROOT = Path(__file__).resolve().parents[2]


@dataclass
class Attribute:
    name: str
    kind: str
    values: list[str] = field(default_factory=list)


@dataclass
class Dataset:
    relation: str
    attributes: list[Attribute]
    rows: list[list[str]]


@dataclass
class Tree:
    prediction: str
    feature_index: Optional[int] = None
    branches: dict[str, "Tree"] = field(default_factory=dict)


def parse_arff(path: Path) -> Dataset:
    relation = path.stem
    attributes: list[Attribute] = []
    rows: list[list[str]] = []
    in_data = False

    try:
        lines = path.read_text(encoding="utf-8-sig").splitlines()
    except OSError as exc:
        raise ValueError(f"Cannot read dataset '{path}': {exc}") from exc

    for line_number, raw_line in enumerate(lines, start=1):
        line = raw_line.strip()
        if not line or line.startswith("%"):
            continue
        if not in_data:
            if line.lower().startswith("@relation"):
                relation = line[len("@relation"):].strip().strip("\"'")
            elif line.lower().startswith("@attribute"):
                match = re.match(
                    r"""@attribute\s+(?:"([^"]+)"|'([^']+)'|(\S+))\s+(.+)""",
                    line,
                    flags=re.IGNORECASE,
                )
                if not match:
                    raise ValueError(f"Invalid @attribute on line {line_number}")
                name = next(group for group in match.groups()[:3] if group is not None)
                declaration = match.group(4).strip()
                if declaration.startswith("{") and declaration.endswith("}"):
                    values = next(csv.reader([declaration[1:-1]], skipinitialspace=True))
                    attributes.append(
                        Attribute(name, "nominal", [value.strip().strip("\"'") for value in values])
                    )
                elif declaration.lower() in {"numeric", "real", "integer"}:
                    attributes.append(Attribute(name, "numeric"))
                else:
                    raise ValueError(
                        f"Unsupported type for attribute '{name}' on line {line_number}: "
                        f"{declaration}"
                    )
            elif line.lower() == "@data":
                in_data = True
                if not attributes:
                    raise ValueError("The ARFF file has no attributes")
        else:
            try:
                row = next(csv.reader([line], skipinitialspace=True))
            except csv.Error as exc:
                raise ValueError(f"Invalid data row on line {line_number}: {exc}") from exc
            row = [value.strip().strip("\"'") for value in row]
            if len(row) != len(attributes):
                raise ValueError(
                    f"Data row on line {line_number} has {len(row)} values; "
                    f"expected {len(attributes)}"
                )
            rows.append(row)

    if not in_data:
        raise ValueError("The ARFF file is missing @data")
    if not rows:
        raise ValueError("The ARFF file contains no data rows")
    return Dataset(relation, attributes, rows)


def split_rows(
    rows: list[list[str]], class_index: int, train_fraction: float, seed: int
) -> tuple[list[list[str]], list[list[str]]]:
    grouped: dict[str, list[list[str]]] = {}
    for row in rows:
        grouped.setdefault(row[class_index], []).append(row)

    rng = random.Random(seed)
    training: list[list[str]] = []
    testing: list[list[str]] = []
    for group in grouped.values():
        rng.shuffle(group)
        train_count = int(len(group) * train_fraction + 0.5)
        train_count = min(max(train_count, 1), len(group))
        training.extend(group[:train_count])
        testing.extend(group[train_count:])
    rng.shuffle(training)
    rng.shuffle(testing)
    return training, testing


def fit_numeric_bins(
    attributes: list[Attribute], training_rows: list[list[str]], class_index: int
) -> tuple[list[Attribute], list[list[str]], dict[int, tuple[float, float, float]]]:
    transformed_attributes = [
        Attribute(attribute.name, attribute.kind, list(attribute.values))
        for attribute in attributes
    ]
    transformed_training = [list(row) for row in training_rows]
    cuts: dict[int, tuple[float, float, float]] = {}

    for index, attribute in enumerate(attributes):
        if attribute.kind != "numeric" or index == class_index:
            continue
        numeric_values = []
        for row in training_rows:
            if row[index] != "?":
                try:
                    number = float(row[index])
                except ValueError as exc:
                    raise ValueError(
                        f"Attribute '{attribute.name}' contains non-numeric value "
                        f"'{row[index]}'"
                    ) from exc
                if not math.isfinite(number):
                    raise ValueError(f"Attribute '{attribute.name}' contains a non-finite value")
                numeric_values.append(number)

        if not numeric_values:
            transformed_attributes[index] = Attribute(
                attribute.name, "nominal", ["bin_1", "bin_2", "bin_3", "?"]
            )
            continue

        minimum, maximum = min(numeric_values), max(numeric_values)
        first_cut = minimum + (maximum - minimum) / 3
        second_cut = minimum + 2 * (maximum - minimum) / 3
        cuts[index] = (minimum, first_cut, second_cut)
        transformed_attributes[index] = Attribute(
            attribute.name, "nominal", ["bin_1", "bin_2", "bin_3", "?"]
        )
        for row in transformed_training:
            row[index] = discretize(row[index], minimum, first_cut, second_cut)

    return transformed_attributes, transformed_training, cuts


def discretize(value: str, minimum: float, first_cut: float, second_cut: float) -> str:
    if value == "?":
        return "?"
    number = float(value)
    if number <= first_cut:
        return "bin_1"
    if number <= second_cut:
        return "bin_2"
    return "bin_3"


def transform_rows(
    rows: list[list[str]], cuts: dict[int, tuple[float, float, float]]
) -> list[list[str]]:
    result = [list(row) for row in rows]
    for index, (minimum, first_cut, second_cut) in cuts.items():
        for row in result:
            row[index] = discretize(row[index], minimum, first_cut, second_cut)
    return result


def entropy(labels: list[str]) -> float:
    total = len(labels)
    counts = Counter(labels)
    return -sum((count / total) * math.log2(count / total) for count in counts.values())


def majority_label(rows: list[list[str]], class_index: int) -> str:
    counts = Counter(row[class_index] for row in rows)
    return max(counts, key=counts.get)


def build_tree(
    rows: list[list[str]], feature_indices: list[int], class_index: int
) -> Tree:
    prediction = majority_label(rows, class_index)
    labels = [row[class_index] for row in rows]
    if len(set(labels)) == 1 or not feature_indices:
        return Tree(prediction)

    base_entropy = entropy(labels)
    best_gain = 0.0
    best_feature: Optional[int] = None
    best_groups: dict[str, list[list[str]]] = {}
    for index in feature_indices:
        groups: dict[str, list[list[str]]] = {}
        for row in rows:
            groups.setdefault(row[index], []).append(row)
        remainder = sum(
            len(group) / len(rows) * entropy([row[class_index] for row in group])
            for group in groups.values()
        )
        gain = base_entropy - remainder
        if gain > best_gain + 1e-12:
            best_gain = gain
            best_feature = index
            best_groups = groups

    if best_feature is None:
        return Tree(prediction)

    remaining = [index for index in feature_indices if index != best_feature]
    return Tree(
        prediction,
        best_feature,
        {
            value: build_tree(group, remaining, class_index)
            for value, group in best_groups.items()
        },
    )


def predict(tree: Tree, row: list[str]) -> str:
    if tree.feature_index is None:
        return tree.prediction
    child = tree.branches.get(row[tree.feature_index])
    return predict(child, row) if child is not None else tree.prediction


def write_arff(path: Path, relation: str, attributes: list[Attribute], rows: list[list[str]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as output:
        output.write(f"@relation {relation}\n\n")
        for index, attribute in enumerate(attributes):
            if attribute.kind == "numeric":
                declaration = "numeric"
            else:
                values = list(attribute.values)
                if any(row[index] == "?" for row in rows) and "?" not in values:
                    values.append("?")
                declaration = "{" + ", ".join(values) + "}"
            output.write(f"@attribute {attribute.name} {declaration}\n")
        output.write("\n@data\n")
        writer = csv.writer(output, lineterminator="\n")
        writer.writerows(rows)


def format_tree(tree: Tree, attributes: list[Attribute], depth: int = 0) -> list[str]:
    indent = "  " * depth
    if tree.feature_index is None:
        return [f"{indent}class: {tree.prediction}"]
    lines: list[str] = []
    feature_name = attributes[tree.feature_index].name
    for value, child in sorted(tree.branches.items()):
        lines.append(f"{indent}{feature_name} = {value}:")
        lines.extend(format_tree(child, attributes, depth + 1))
    return lines


def run(args: argparse.Namespace) -> float:
    dataset = parse_arff(args.dataset)
    class_index = (
        next(
            (i for i, attribute in enumerate(dataset.attributes)
             if attribute.name.lower() == args.class_attribute.lower()),
            -1,
        )
        if args.class_attribute
        else len(dataset.attributes) - 1
    )
    if class_index < 0:
        raise ValueError(f"Class attribute '{args.class_attribute}' was not found")

    if len(dataset.rows) < 2:
        raise ValueError("At least two data rows are required for a train/test split")

    raw_training, raw_testing = split_rows(
        dataset.rows, class_index, args.train_fraction, args.seed
    )
    if not raw_testing:
        raw_testing.append(raw_training.pop())

    attributes, training, cuts = fit_numeric_bins(
        dataset.attributes, raw_training, class_index
    )
    testing = transform_rows(raw_testing, cuts)
    feature_indices = [i for i in range(len(attributes)) if i != class_index]
    tree = build_tree(training, feature_indices, class_index)
    correct = sum(predict(tree, row) == row[class_index] for row in testing)
    accuracy = correct / len(testing)

    output_dir = args.output_dir or PROJECT_ROOT / "results" / args.dataset.stem
    output_dir.mkdir(parents=True, exist_ok=True)
    write_arff(output_dir / "training_set.arff", dataset.relation, attributes, training)
    write_arff(output_dir / "testing_set.arff", dataset.relation, attributes, testing)
    (output_dir / "decision_tree.txt").write_text(
        "\n".join(format_tree(tree, attributes)) + "\n", encoding="utf-8"
    )
    (output_dir / "evaluation.txt").write_text(
        f"Training samples: {len(training)}\n"
        f"Testing samples: {len(testing)}\n"
        f"Correctly classified: {correct}\n"
        f"Accuracy: {accuracy:.2%}\n"
        f"Random seed: {args.seed}\n",
        encoding="utf-8",
    )
    print(f"Training samples: {len(training)}")
    print(f"Testing samples: {len(testing)}")
    print(f"Correctly classified: {correct}")
    print(f"Accuracy: {accuracy:.2%}")
    print(f"Results written to: {output_dir}")
    return accuracy


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Discretize numeric ARFF features, train an ID3 tree, and report test accuracy."
    )
    parser.add_argument(
        "dataset",
        type=Path,
        nargs="?",
        default=PROJECT_ROOT / "decision_tree" / "data" / "soybean_modify.arff",
        help="Input ARFF file (default: decision_tree/data/soybean_modify.arff)",
    )
    parser.add_argument("--train-fraction", type=float, default=0.7)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--class-attribute", help="Target attribute (default: last attribute)")
    parser.add_argument("--output-dir", type=Path, help="Output directory")
    args = parser.parse_args()
    if not 0 < args.train_fraction < 1:
        parser.error("--train-fraction must be greater than 0 and less than 1")
    try:
        accuracy = run(args)
    except ValueError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2
    print(f"Accuracy: {accuracy:.2%}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
