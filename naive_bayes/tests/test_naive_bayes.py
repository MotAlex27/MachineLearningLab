import tempfile
import unittest
from pathlib import Path

from naive_bayes.src.naive_bayes import Document, fit, parse_reuters


class NaiveBayesTests(unittest.TestCase):
    def test_parser_reads_sparse_vectors_and_uses_first_topic(self):
        contents = (
            "#Samples 2\n"
            "#Attributes 3\n"
            "#Topics 2\n"
            "@attribute 10 1.0\n"
            "@attribute 20 2.0\n"
            "@attribute 30 3.0\n"
            "@topic c1 1\n"
            "@topic c2 1\n"
            "@data\n"
            "0:4 2:1 # c1 c2\n"
            "1:0 # c2\n"
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "sample.arff"
            path.write_text(contents, encoding="utf-8")
            dataset = parse_reuters(path)

        self.assertEqual(dataset.attributes, ("10", "20", "30"))
        self.assertEqual(
            dataset.documents,
            (Document(frozenset({0, 2}), "c1"), Document(frozenset(), "c2")),
        )

    def test_bernoulli_model_learns_presence_and_absence(self):
        training = (
            Document(frozenset({0}), "a"),
            Document(frozenset({0}), "a"),
            Document(frozenset({1}), "b"),
            Document(frozenset({1}), "b"),
        )
        model = fit(training, feature_count=2)

        self.assertEqual(model.predict(frozenset({0})), "a")
        self.assertEqual(model.predict(frozenset({1})), "b")

    def test_unknown_features_and_invalid_smoothing_are_rejected(self):
        model = fit((Document(frozenset({0}), "a"),), feature_count=1)
        with self.assertRaises(ValueError):
            model.predict(frozenset({1}))
        with self.assertRaises(ValueError):
            fit((Document(frozenset(), "a"),), feature_count=1, alpha=0)


if __name__ == "__main__":
    unittest.main()
