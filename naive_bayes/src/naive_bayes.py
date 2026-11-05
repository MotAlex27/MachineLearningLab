#!/usr/bin/env python3
"""Train and evaluate a Bernoulli Naive Bayes classifier on Reuters vectors."""

from __future__ import annotations

import argparse
import csv
import math
import re
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

PROJECT_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class Document:
    features: frozenset[int]
    label: str


@dataclass(frozen=True)
class ReutersDataset:
    attributes: tuple[str, ...]
    documents: tuple[Document, ...]


@dataclass
class BernoulliNaiveBayes:
    feature_count: int
    classes: tuple[str, ...]
    log_priors: dict[str, float]
    log_absent: dict[str, float]
    log_present_delta: dict[str, list[float]]

    def predict(self, features: frozenset[int]) -> str:
        if any(index < 0 or index >= self.feature_count for index in features):
            raise ValueError("A document contains a feature index outside the model vocabulary")
        best_class = self.classes[0]
        best_score = float("-inf")
        for label in self.classes:
            score = self.log_priors[label] + self.log_absent[label]
            score += sum(self.log_present_delta[label][index] for index in features)
            if score > best_score:
                best_class = label
                best_score = score
        return best_class


def parse_reuters(path: Path) -> ReutersDataset:
    try:
        lines = path.read_text(encoding="utf-8-sig").splitlines()
    except OSError as exc:
        raise ValueError(f"Cannot read dataset '{path}': {exc}") from exc

    declared_samples: Optional[int] = None
    declared_attributes: Optional[int] = None
    attributes: list[str] = []
    documents: list[Document] = []
    in_data = False

    for line_number, raw_line in enumerate(lines, start=1):
        line = raw_line.strip()
        if not line:
            continue
        if not in_data:
            if line.startswith("#"):
                metadata = re.fullmatch(r"#\s*(Samples|Attributes|Topics)\s+(\d+)", line, re.I)
                if metadata:
                    key, value = metadata.group(1).lower(), int(metadata.group(2))
                    if key == "samples":
                        declared_samples = value
                    elif key == "attributes":
                        declared_attributes = value
                continue
            if line.lower().startswith("@attribute"):
                match = re.fullmatch(r"@attribute\s+(\S+)(?:\s+\S+)?", line, re.I)
                if not match:
                    raise ValueError(f"Invalid @attribute on line {line_number} in '{path}'")
                name = match.group(1)
                if name in attributes:
                    raise ValueError(f"Duplicate attribute '{name}' on line {line_number} in '{path}'")
                attributes.append(name)
            elif line.lower().startswith("@topic"):
                continue
            elif line.lower() == "@data":
                in_data = True
                if not attributes:
                    raise ValueError(f"The Reuters dataset '{path}' has no attributes")
                if declared_attributes is not None and declared_attributes != len(attributes):
                    raise ValueError(
                        f"Header declares {declared_attributes} attributes but lists "
                        f"{len(attributes)} in '{path}'"
                    )
            else:
                raise ValueError(f"Unexpected header on line {line_number} in '{path}': {line}")
            continue

        vector, separator, topic_text = line.partition("#")
        if not separator:
            raise ValueError(f"Missing topic marker '#' on data line {line_number} in '{path}'")
        topics = topic_text.split()
        if not topics:
            raise ValueError(f"Missing topic on data line {line_number} in '{path}'")
        features: set[int] = set()
        for pair in vector.replace(",", " ").split():
            match = re.fullmatch(r"(\d+):(\d+)", pair)
            if not match:
                raise ValueError(f"Invalid feature '{pair}' on data line {line_number} in '{path}'")
            index, frequency = int(match.group(1)), int(match.group(2))
            if index >= len(attributes):
                raise ValueError(
                    f"Feature index {index} is outside the {len(attributes)} attributes "
                    f"on data line {line_number} in '{path}'"
                )
            if frequency:
                features.add(index)
        documents.append(Document(frozenset(features), topics[0]))

    if not in_data:
        raise ValueError(f"The Reuters dataset '{path}' is missing @data")
    if not documents:
        raise ValueError(f"The Reuters dataset '{path}' contains no documents")
    if declared_samples is not None and declared_samples != len(documents):
        raise ValueError(
            f"Header declares {declared_samples} samples but contains {len(documents)} "
            f"documents in '{path}'"
        )
    return ReutersDataset(tuple(attributes), tuple(documents))


def fit(
    documents: tuple[Document, ...], feature_count: int, alpha: float = 1.0
) -> BernoulliNaiveBayes:
    if not documents:
        raise ValueError("At least one training document is required")
    if feature_count <= 0:
        raise ValueError("The vocabulary must contain at least one feature")
    if not math.isfinite(alpha) or alpha <= 0:
        raise ValueError("Smoothing alpha must be a finite number greater than zero")

    class_counts = Counter(document.label for document in documents)
    classes = tuple(class_counts)
    feature_counts = {label: [0] * feature_count for label in classes}
    for document in documents:
        if any(index < 0 or index >= feature_count for index in document.features):
            raise ValueError("A training document contains a feature index outside the vocabulary")
        for index in document.features:
            feature_counts[document.label][index] += 1

    log_priors: dict[str, float] = {}
    log_absent: dict[str, float] = {}
    log_present_delta: dict[str, list[float]] = {}
    for label in classes:
        log_priors[label] = math.log(class_counts[label] / len(documents))
        probabilities = [
            (count + alpha) / (class_counts[label] + 2 * alpha)
            for count in feature_counts[label]
        ]
        log_absent[label] = sum(math.log1p(-probability) for probability in probabilities)
        log_present_delta[label] = [
            math.log(probability) - math.log1p(-probability)
            for probability in probabilities
        ]

    return BernoulliNaiveBayes(
        feature_count, classes, log_priors, log_absent, log_present_delta
    )


def run(training_path: Path, testing_path: Path, output_dir: Path, alpha: float) -> float:
    training = parse_reuters(training_path)
    testing = parse_reuters(testing_path)
    if training.attributes != testing.attributes:
        raise ValueError("Training and testing files have different attribute lists")

    model = fit(training.documents, len(training.attributes), alpha)
    predictions = [
        (document.label, model.predict(document.features))
        for document in testing.documents
    ]
    correct = sum(actual == predicted for actual, predicted in predictions)
    accuracy = correct / len(predictions)
    class_counts = Counter(actual for actual, _ in predictions)
    true_positives = Counter(
        actual for actual, predicted in predictions if actual == predicted
    )
    predicted_counts = Counter(predicted for _, predicted in predictions)

    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "predictions.csv").open("w", encoding="utf-8", newline="") as output:
        writer = csv.writer(output)
        writer.writerow(["actual", "predicted", "correct"])
        writer.writerows(
            (actual, predicted, actual == predicted) for actual, predicted in predictions
        )

    evaluation = [
        f"Training samples: {len(training.documents)}",
        f"Testing samples: {len(testing.documents)}",
        f"Attributes: {len(training.attributes)}",
        f"Classes: {len(model.classes)}",
        f"Correctly classified: {correct}",
        f"Accuracy: {accuracy:.2%}",
        f"Laplace smoothing alpha: {alpha:g}",
        "",
        "Per-class metrics (class, support, precision, recall, F1):",
    ]
    for label in sorted(class_counts):
        precision = true_positives[label] / predicted_counts[label] if predicted_counts[label] else 0
        recall = true_positives[label] / class_counts[label]
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0
        evaluation.append(
            f"{label}, {class_counts[label]}, {precision:.4f}, {recall:.4f}, {f1:.4f}"
        )
    (output_dir / "evaluation.txt").write_text("\n".join(evaluation) + "\n", encoding="utf-8")

    print(f"Training samples: {len(training.documents)}")
    print(f"Testing samples: {len(testing.documents)}")
    print(f"Attributes: {len(training.attributes)}")
    print(f"Correctly classified: {correct}")
    print(f"Accuracy: {accuracy:.2%}")
    print(f"Results written to: {output_dir}")
    return accuracy


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Train and evaluate Bernoulli Naive Bayes on sparse Reuters documents."
    )
    parser.add_argument(
        "training_file",
        type=Path,
        nargs="?",
        default=(
            PROJECT_ROOT
            / "naive_bayes"
            / "data"
            / "Reuters_Dataset"
            / "MultiClass_IG_Training_Chose_1309.0.arff"
        ),
        help="Reuters training file",
    )
    parser.add_argument(
        "testing_file",
        type=Path,
        nargs="?",
        default=(
            PROJECT_ROOT
            / "naive_bayes"
            / "data"
            / "Reuters_Dataset"
            / "MultiClass_IG_Testing_Chose_1309.0.arff"
        ),
        help="Reuters testing file",
    )
    parser.add_argument("--alpha", type=float, default=1.0, help="Positive Laplace smoothing value")
    parser.add_argument("--output-dir", type=Path, help="Directory for evaluation results")
    args = parser.parse_args()
    output_dir = args.output_dir or PROJECT_ROOT / "results" / f"naive_bayes_{args.training_file.stem}"
    try:
        run(args.training_file, args.testing_file, output_dir, args.alpha)
    except ValueError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
