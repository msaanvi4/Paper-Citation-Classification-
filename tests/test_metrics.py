import json
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
METRICS_FILE = PROJECT_ROOT / "results" / "metrics.json"


class TestEvaluationMetrics(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        with METRICS_FILE.open(encoding="utf-8") as file:
            cls.data = json.load(file)

    def test_test_sample_count(self):
        self.assertEqual(self.data["test_samples"], 18000)

    def test_all_models_are_present(self):
        expected = {"Naive Bayes", "Random Forest", "Neural Network"}
        self.assertEqual(set(self.data["models"].keys()), expected)

    def test_metrics_are_within_valid_range(self):
        metric_names = {
            "accuracy",
            "weighted_precision",
            "weighted_recall",
            "weighted_f1",
        }

        for model_name, metrics in self.data["models"].items():
            with self.subTest(model=model_name):
                self.assertTrue(metric_names.issubset(metrics.keys()))

                for metric in metric_names:
                    self.assertGreaterEqual(metrics[metric], 0)
                    self.assertLessEqual(metrics[metric], 1)


if __name__ == "__main__":
    unittest.main()
