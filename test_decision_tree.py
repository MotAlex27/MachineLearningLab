import tempfile
import unittest
from pathlib import Path

from decision_tree import (
    Attribute,
    build_tree,
    fit_numeric_bins,
    parse_arff,
    predict,
    split_rows,
    transform_rows,
)


class DecisionTreeTests(unittest.TestCase):
    def test_numeric_features_are_binned_using_training_data(self):
        attributes = [Attribute("measurement", "numeric"), Attribute("class", "nominal", ["a", "b"])]
        training = [["0", "a"], ["3", "a"], ["6", "b"], ["9", "b"]]
        transformed_attributes, transformed_training, cuts = fit_numeric_bins(
            attributes, training, 1
        )

        self.assertEqual(
            transformed_attributes[0].values, ["bin_1", "bin_2", "bin_3", "?"]
        )
        self.assertEqual(
            [row[0] for row in transformed_training],
            ["bin_1", "bin_1", "bin_2", "bin_3"],
        )
        self.assertEqual(
            [row[0] for row in transform_rows([["100", "b"], ["?", "a"]], cuts)],
            ["bin_3", "?"],
        )

    def test_id3_learns_and_predicts_nominal_rules(self):
        rows = [
            ["sunny", "no"],
            ["sunny", "no"],
            ["rainy", "yes"],
            ["rainy", "yes"],
        ]
        tree = build_tree(rows, [0], 1)
        self.assertEqual(predict(tree, ["sunny"]), "no")
        self.assertEqual(predict(tree, ["rainy"]), "yes")
        self.assertEqual(predict(tree, ["unseen"]), tree.prediction)

    def test_split_is_deterministic_and_preserves_rows(self):
        rows = [[str(index), "yes" if index % 2 else "no"] for index in range(100)]
        train_a, test_a = split_rows(rows, 1, 0.7, 42)
        train_b, test_b = split_rows(rows, 1, 0.7, 42)
        self.assertEqual((train_a, test_a), (train_b, test_b))
        self.assertEqual(len(train_a), 70)
        self.assertEqual(len(test_a), 30)

    def test_arff_nominal_codes_remain_nominal(self):
        contents = (
            "@relation sample\n"
            "@attribute coded {0, 1, 2}\n"
            "@attribute class {a, b}\n"
            "@data\n"
            "0,a\n"
            "2,b\n"
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "sample.arff"
            path.write_text(contents, encoding="utf-8")
            dataset = parse_arff(path)
        self.assertEqual(dataset.attributes[0].kind, "nominal")
        self.assertEqual(dataset.attributes[0].values, ["0", "1", "2"])


if __name__ == "__main__":
    unittest.main()
