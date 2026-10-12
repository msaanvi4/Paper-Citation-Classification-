"""Predict Low/Medium/High citation category for a new paper."""
from pathlib import Path
import argparse
import pickle
import pandas as pd
from src.train import MODEL_PATH


def predict_paper(title, abstract, publication_year, author_count, primary_field):
    if not MODEL_PATH.exists():
        raise FileNotFoundError("Trained model not found. Run `python -m src.train` first.")
    record = pd.DataFrame([{
        "combined_text": f"{title or ''} {abstract or ''}".strip(),
        "publication_year": float(publication_year),
        "author_count": float(author_count),
        "primary_field": primary_field or "Unknown",
    }])
    with MODEL_PATH.open("rb") as file:
        model = pickle.load(file)
    predicted_class = str(model.predict(record)[0])
    confidence = None
    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(record)[0]
        confidence = float(max(probabilities))
    return {"citation_category": predicted_class, "confidence": confidence}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--title", required=True)
    parser.add_argument("--abstract", default="")
    parser.add_argument("--year", required=True, type=float)
    parser.add_argument("--authors", required=True, type=float)
    parser.add_argument("--field", required=True)
    args = parser.parse_args()
    result = predict_paper(args.title, args.abstract, args.year, args.authors, args.field)
    print(f"Predicted citation category: {result['citation_category']}")
    if result["confidence"] is not None:
        print(f"Model confidence (not a guarantee): {result['confidence']:.1%}")


if __name__ == "__main__":
    main()
