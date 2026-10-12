"""Data loading, validation, target creation, and train/test splitting."""
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET = PROJECT_ROOT / "dataset" / "dataset.csv"

REQUIRED_COLUMNS = [
    "title", "abstract", "publication_year", "author_count",
    "primary_field", "citation_count",
]
FEATURE_COLUMNS = ["combined_text", "publication_year", "author_count", "primary_field"]
TARGET_COLUMN = "citation_category"


def citation_category(citation_count):
    """Map citation counts to the project's Low/Medium/High classes."""
    if pd.isna(citation_count):
        raise ValueError("citation_count contains missing values.")
    value = float(citation_count)
    if value <= 918:
        return "Low"
    if value <= 1384:
        return "Medium"
    return "High"


def load_and_prepare_data(csv_path=DEFAULT_DATASET):
    """Load the CSV, validate its schema, clean required fields, and add labels."""
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {csv_path}. Put the real dataset at dataset/dataset.csv."
        )

    df = pd.read_csv(csv_path)
    missing_columns = [column for column in REQUIRED_COLUMNS if column not in df.columns]
    if missing_columns:
        raise ValueError(f"Dataset is missing required columns: {missing_columns}")

    df = df.copy()
    df["title"] = df["title"].fillna("").astype(str)
    df["abstract"] = df["abstract"].fillna("").astype(str)
    df["primary_field"] = df["primary_field"].fillna("Unknown").astype(str)
    df["publication_year"] = pd.to_numeric(df["publication_year"], errors="coerce")
    df["author_count"] = pd.to_numeric(df["author_count"], errors="coerce")
    df["citation_count"] = pd.to_numeric(df["citation_count"], errors="coerce")

    df = df.dropna(subset=["publication_year", "author_count", "citation_count"])
    df = df[df["citation_count"] >= 0].copy()
    if df.empty:
        raise ValueError("No usable rows remain after cleaning the dataset.")

    df["combined_text"] = (df["title"].str.strip() + " " + df["abstract"].str.strip()).str.strip()
    df[TARGET_COLUMN] = df["citation_count"].map(citation_category)
    return df


def split_data(df, test_size=0.15, random_state=42):
    """Create a reproducible stratified train/test split."""
    X = df[FEATURE_COLUMNS].copy()
    y = df[TARGET_COLUMN].copy()
    counts = y.value_counts()
    if len(counts) < 2 or counts.min() < 2:
        raise ValueError("Need at least two classes and at least two examples per class.")
    return train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
