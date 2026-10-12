"""Evaluate the saved model and write metrics/report/confusion matrix."""
from pathlib import Path
import json
import pickle
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    classification_report, confusion_matrix, ConfusionMatrixDisplay,
)
from src.preprocessing import DEFAULT_DATASET, load_and_prepare_data, split_data
from src.train import MODEL_PATH

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = PROJECT_ROOT / "results"


def main():
    if not MODEL_PATH.exists():
        raise FileNotFoundError("Trained model not found. Run `python -m src.train` first.")

    df = load_and_prepare_data(DEFAULT_DATASET)
    _, X_test, _, y_test = split_data(df)
    with MODEL_PATH.open("rb") as file:
        model = pickle.load(file)
    predictions = model.predict(X_test)

    precision, recall, f1, _ = precision_recall_fscore_support(
        y_test, predictions, average="weighted", zero_division=0
    )
    metrics = {
        "accuracy": float(accuracy_score(y_test, predictions)),
        "weighted_precision": float(precision),
        "weighted_recall": float(recall),
        "weighted_f1": float(f1),
        "test_samples": int(len(y_test)),
        "classes": [str(c) for c in model.classes_],
    }

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / "metrics.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"
    )
    (RESULTS_DIR / "classification_report.txt").write_text(
        classification_report(y_test, predictions, zero_division=0),
        encoding="utf-8",
    )

    labels = ["Low", "Medium", "High"]
    matrix = confusion_matrix(y_test, predictions, labels=labels)
    display = ConfusionMatrixDisplay(confusion_matrix=matrix, display_labels=labels)
    display.plot(cmap="Blues", values_format="d")
    plt.title("Citation Category Confusion Matrix")
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "confusion_matrix.png", dpi=200)
    plt.close()

    print(json.dumps(metrics, indent=2))
    print(f"Saved evaluation files in: {RESULTS_DIR}")


if __name__ == "__main__":
    main()
