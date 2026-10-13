
"""Tests for dataset preprocessing and citation classification."""

import tempfile
import unittest
from pathlib import Path

import pandas as pd

from src.preprocessing import (
    FEATURE_COLUMNS,
    TARGET_COLUMN,
    citation_category,
    load_and_prepare_data,
    split_data,
)


class TestCitationCategory(unittest.TestCase):
    """Test citation-category labels and boundary values."""

    def test_low_category(self):
        self.assertEqual(citation_category(0), "Low")
        self.assertEqual(citation_category(918), "Low")

    def test_medium_category(self):
        self.assertEqual(citation_category(919), "Medium")
        self.assertEqual(citation_category(1384), "Medium")

    def test_high_category(self):
        self.assertEqual(citation_category(1385), "High")
        self.assertEqual(citation_category(5000), "High")

    def test_missing_citation_count_raises_error(self):
        with self.assertRaises(ValueError):
            citation_category(float("nan"))


class TestLoadAndPrepareData(unittest.TestCase):
    """Test dataset validation and cleaning."""

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.csv_path = Path(self.temp_dir.name) / "dataset.csv"

        self.data = pd.DataFrame({
            "paper_id": [1, 2, 3, 4, 5, 6],
            "title": [
                "Paper A",
                None,
                "Paper C",
                "Paper D",
                "Paper E",
                "Paper F",
            ],
            "abstract": [
                "Abstract A",
                "Abstract B",
                None,
                "Abstract D",
                "Abstract E",
                "Abstract F",
            ],
            "publication_year": [
                2020, 2021, 2022, 2023, 2024, 2025
            ],
            "author_count": [2, 3, 4, 2, 5, 1],
            "primary_field": [
                "AI", "DBMS", None, "ML", "AI", "Systems"
            ],
            "citation_count": [100, 918, 919, 1384, 1385, 200],
        })

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_missing_text_values_are_filled(self):
        self.data.to_csv(self.csv_path, index=False)

        result = load_and_prepare_data(self.csv_path)

        self.assertEqual(result.loc[1, "title"], "")
        self.assertEqual(result.loc[2, "abstract"], "")
        self.assertEqual(result.loc[2, "primary_field"], "Unknown")

    def test_combined_text_is_created(self):
        self.data.to_csv(self.csv_path, index=False)

        result = load_and_prepare_data(self.csv_path)

        self.assertEqual(
            result.loc[0, "combined_text"],
            "Paper A Abstract A",
        )

    def test_citation_categories_are_created(self):
        self.data.to_csv(self.csv_path, index=False)

        result = load_and_prepare_data(self.csv_path)

        self.assertEqual(
            result[TARGET_COLUMN].tolist(),
            ["Low", "Low", "Medium", "Medium", "High", "Low"],
        )

    def test_invalid_numeric_rows_are_removed(self):
        # Use object dtype so the test can insert invalid text.
        self.data["publication_year"] = (
            self.data["publication_year"].astype(object)
        )
        self.data.loc[1, "publication_year"] = "invalid"
        self.data.loc[2, "author_count"] = None
        self.data.loc[3, "citation_count"] = -1

        self.data.to_csv(self.csv_path, index=False)
        result = load_and_prepare_data(self.csv_path)

        # Rows 1, 2, and 3 should be removed.
        self.assertEqual(len(result), 3)
        self.assertTrue((result["citation_count"] >= 0).all())
        self.assertTrue(result["publication_year"].notna().all())
        self.assertTrue(result["author_count"].notna().all())

    def test_missing_required_column_raises_error(self):
        self.data.drop(columns=["title"]).to_csv(
            self.csv_path,
            index=False,
        )

        with self.assertRaisesRegex(
            ValueError,
            "missing required columns",
        ):
            load_and_prepare_data(self.csv_path)

    def test_missing_dataset_raises_error(self):
        missing_path = Path(self.temp_dir.name) / "missing.csv"

        with self.assertRaises(FileNotFoundError):
            load_and_prepare_data(missing_path)

    def test_empty_cleaned_dataset_raises_error(self):
        self.data["citation_count"] = -1
        self.data.to_csv(self.csv_path, index=False)

        with self.assertRaisesRegex(ValueError, "No usable records"):
            load_and_prepare_data(self.csv_path)


class TestSplitData(unittest.TestCase):
    """Test train/test split behavior."""

    def setUp(self):
        self.df = pd.DataFrame({
            "combined_text": [
                f"Paper text {i}" for i in range(30)
            ],
            "publication_year": [
                2020 + i % 5 for i in range(30)
            ],
            "author_count": [
                1 + i % 4 for i in range(30)
            ],
            "primary_field": ["AI", "DBMS", "ML"] * 10,
            TARGET_COLUMN: ["Low", "Medium", "High"] * 10,
        })

    def test_split_sizes_and_columns(self):
        X_train, X_test, y_train, y_test = split_data(self.df)

        self.assertEqual(len(X_train), 25)
        self.assertEqual(len(X_test), 5)
        self.assertEqual(X_train.columns.tolist(), FEATURE_COLUMNS)
        self.assertEqual(len(X_train), len(y_train))
        self.assertEqual(len(X_test), len(y_test))

    def test_split_is_reproducible(self):
        first = split_data(self.df)
        second = split_data(self.df)

        for first_part, second_part in zip(first, second):
            if isinstance(first_part, pd.DataFrame):
                pd.testing.assert_frame_equal(first_part, second_part)
            else:
                pd.testing.assert_series_equal(first_part, second_part)

    def test_too_few_class_samples_raises_error(self):
        small_df = self.df.copy()
        small_df[TARGET_COLUMN] = ["Low"] * 29 + ["High"]

        with self.assertRaisesRegex(
            ValueError,
            "at least two classes",
        ):
            split_data(small_df)


if __name__ == "__main__":
    unittest.main()
