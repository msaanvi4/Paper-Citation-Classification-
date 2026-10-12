"""Train and save the citation-category classifier."""
from pathlib import Path
import pickle
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import make_column_selector
from sklearn.pipeline import FeatureUnion
from sklearn.preprocessing import FunctionTransformer

from src.preprocessing import DEFAULT_DATASET, load_and_prepare_data, split_data, FEATURE_COLUMNS

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = PROJECT_ROOT / "models" / "trained_model.pkl"


def build_model():
    """Build one reproducible pipeline that includes fitted preprocessing."""
    text_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="constant", fill_value="")),
        ("flatten", FunctionTransformer(lambda x: x.ravel(), validate=False)),
        ("tfidf", TfidfVectorizer(
            max_features=20000, ngram_range=(1, 2), stop_words="english",
            min_df=1, max_df=1.0, sublinear_tf=True
        )),
    ])
    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])
    features = ColumnTransformer([
        ("text", text_pipeline, ["combined_text"]),
        ("numeric", numeric_pipeline, ["publication_year", "author_count"]),
        ("categorical", categorical_pipeline, ["primary_field"]),
    ])
    return Pipeline([
        ("features", features),
        ("classifier", LogisticRegression(max_iter=1000, class_weight="balanced")),
    ])


def main():
    df = load_and_prepare_data(DEFAULT_DATASET)
    X_train, X_test, y_train, y_test = split_data(df)
    model = build_model()
    model.fit(X_train, y_train)

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    with MODEL_PATH.open("wb") as file:
        pickle.dump(model, file)

    print(f"Training rows: {len(X_train):,}")
    print(f"Validation/test rows: {len(X_test):,}")
    print(f"Classes: {sorted(y_train.unique())}")
    print(f"Saved trained pipeline to: {MODEL_PATH}")


if __name__ == "__main__":
    main()
