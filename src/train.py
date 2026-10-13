
"""Train three paper citation classification models."""

from pathlib import Path
import pickle

import numpy as np

from sklearn.compose import ColumnTransformer
from sklearn.decomposition import TruncatedSVD
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.naive_bayes import MultinomialNB
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.preprocessing import (
    DEFAULT_DATASET,
    load_and_prepare_data,
    split_data,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = PROJECT_ROOT / "models" / "trained_model.pkl"


def build_text_pipeline():
    """Convert the combined text column into TF-IDF features."""
    return Pipeline([
        (
            "tfidf",
            TfidfVectorizer(
                max_features=20000,
                ngram_range=(1, 2),
                stop_words="english",
                min_df=3,
                max_df=0.95,
                sublinear_tf=True,
            ),
        ),
    ])


def build_metadata_pipeline():
    """Preprocess numeric and categorical metadata."""
    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])

    return ColumnTransformer([
        (
            "numeric",
            numeric_pipeline,
            ["publication_year", "author_count"],
        ),
        (
            "categorical",
            categorical_pipeline,
            ["primary_field"],
        ),
    ])


def build_combined_features():
    """Combine TF-IDF text features with paper metadata."""
    return ColumnTransformer([
        (
            "text",
            build_text_pipeline(),
            "combined_text",
        ),
        (
            "metadata",
            build_metadata_pipeline(),
            [
                "publication_year",
                "author_count",
                "primary_field",
            ],
        ),
    ])


def build_naive_bayes_pipeline():
    """Train Naive Bayes using only text features."""
    text_features = ColumnTransformer([
        (
            "text",
            build_text_pipeline(),
            "combined_text",
        ),
    ])

    return Pipeline([
        ("features", text_features),
        ("classifier", MultinomialNB()),
    ])


def build_random_forest_pipeline():
    """Train Random Forest using text and metadata."""
    return Pipeline([
        ("features", build_combined_features()),
        (
            "classifier",
            RandomForestClassifier(
                n_estimators=100,
                max_depth=20,
                min_samples_leaf=5,
                random_state=42,
                n_jobs=-1,
            ),
        ),
    ])


def build_neural_network_pipeline():
    """Train an MLP using reduced text features and metadata."""
    text_pipeline = Pipeline([
        (
            "tfidf",
            TfidfVectorizer(
                max_features=20000,
                ngram_range=(1, 2),
                stop_words="english",
                min_df=3,
                max_df=0.95,
                sublinear_tf=True,
            ),
        ),
        (
            "svd",
            TruncatedSVD(
                n_components=200,
                random_state=42,
            ),
        ),
    ])

    features = ColumnTransformer([
        (
            "text",
            text_pipeline,
            "combined_text",
        ),
        (
            "metadata",
            build_metadata_pipeline(),
            [
                "publication_year",
                "author_count",
                "primary_field",
            ],
        ),
    ])

    return Pipeline([
        ("features", features),
        (
            "classifier",
            MLPClassifier(
                hidden_layer_sizes=(64,),
                activation="relu",
                solver="adam",
                learning_rate_init=0.001,
                batch_size=256,
                max_iter=15,
                early_stopping=True,
                random_state=42,
            ),
        ),
    ])


def main():
    """Load the dataset, train all models, and save the pipelines."""
    df = load_and_prepare_data(DEFAULT_DATASET)
    X_train, X_test, y_train, y_test = split_data(df)

    models = {
        "Naive Bayes": build_naive_bayes_pipeline(),
        "Random Forest": build_random_forest_pipeline(),
        "Neural Network": build_neural_network_pipeline(),
    }

    trained_models = {}

    for name, model in models.items():
        print(f"\nTraining {name}...", flush=True)

        model.fit(X_train, y_train)
        accuracy = model.score(X_test, y_test)

        print(f"{name} test accuracy: {accuracy:.4f}", flush=True)
        trained_models[name] = model

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)

    with MODEL_PATH.open("wb") as file:
        pickle.dump(trained_models, file)

    print(f"\nTraining rows: {len(X_train):,}")
    print(f"Test rows: {len(X_test):,}")
    print(f"Classes: {sorted(y_train.unique())}")
    print(f"Saved all three models to: {MODEL_PATH}")


if __name__ == "__main__":
    main()
