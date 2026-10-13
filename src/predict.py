
"""Predict the citation category of a new research paper."""

import argparse
import pickle

import pandas as pd

from src.train import MODEL_PATH


AVAILABLE_MODELS = (
    "Naive Bayes",
    "Random Forest",
    "Neural Network",
)


def predict_paper(
    title,
    abstract,
    publication_year,
    author_count,
    primary_field,
    model_name="Neural Network",
):
    """Predict Low, Medium, or High citations using a selected model."""

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Trained model not found: {MODEL_PATH}. "
            "Run 'python -m src.train' first."
        )

    if model_name not in AVAILABLE_MODELS:
        raise ValueError(
            f"Unknown model: {model_name}. "
            f"Choose one of: {', '.join(AVAILABLE_MODELS)}"
        )

    record = pd.DataFrame([{
        "combined_text": (
            f"{title or ''} {abstract or ''}"
        ).strip(),
        "publication_year": float(publication_year),
        "author_count": float(author_count),
        "primary_field": (primary_field or "Unknown").strip() or "Unknown",
    }])

    with MODEL_PATH.open("rb") as file:
        saved_models = pickle.load(file)

    if not isinstance(saved_models, dict):
        raise TypeError(
            "Expected a dictionary of trained models. "
            "Run 'python -m src.train' using the updated training code."
        )

    if model_name not in saved_models:
        raise KeyError(
            f"Model '{model_name}' was not found in the saved model file. "
            f"Available models: {', '.join(saved_models.keys())}"
        )

    model = saved_models[model_name]
    predicted_class = str(model.predict(record)[0])

    confidence = None

    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(record)[0]
        confidence = float(max(probabilities))

    return {
        "citation_category": predicted_class,
        "confidence": confidence,
        "model": model_name,
    }


def main():
    """Read command-line arguments and print the prediction."""

    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument("--title", required=True)
    parser.add_argument("--abstract", default="")
    parser.add_argument("--year", required=True, type=float)
    parser.add_argument("--authors", required=True, type=float)
    parser.add_argument("--field", required=True)

    parser.add_argument(
        "--model",
        choices=AVAILABLE_MODELS,
        default="Neural Network",
        help="Model to use (default: Neural Network).",
    )

    args = parser.parse_args()

    if args.year <= 0:
        parser.error("--year must be greater than zero.")

    if args.authors <= 0:
        parser.error("--authors must be greater than zero.")

    result = predict_paper(
        title=args.title,
        abstract=args.abstract,
        publication_year=args.year,
        author_count=args.authors,
        primary_field=args.field,
        model_name=args.model,
    )

    print(f"Model: {result['model']}")
    print(f"Predicted citation category: {result['citation_category']}")

    if result["confidence"] is not None:
        print(
            "Model confidence (not a guarantee): "
            f"{result['confidence']:.1%}"
        )


if __name__ == "__main__":
    main()
