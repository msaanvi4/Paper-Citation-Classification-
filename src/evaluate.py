
"""Evaluate all trained paper citation classification models."""

import json
import pickle
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    precision_recall_fscore_support,
)

from src.preprocessing import DEFAULT_DATASET, load_and_prepare_data, split_data


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = PROJECT_ROOT / "models" / "trained_model.pkl"
RESULTS_DIR = PROJECT_ROOT / "results"
METRICS_PATH = RESULTS_DIR / "metrics.json"
REPORT_PATH = RESULTS_DIR / "classification_report.txt"
CONFUSION_MATRIX_PATH = RESULTS_DIR / "confusion_matrix.png"


def main():
    """Evaluate each saved model on the same held-out test set."""
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Trained model file not found: {MODEL_PATH}\n"
            "Run 'python -m src.train' first."
        )

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # Recreate the same deterministic train/test split used during training.
    df = load_and_prepare_data(DEFAULT_DATASET)
    _, X_test, _, y_test = split_data(df)

    with MODEL_PATH.open("rb") as file:
        models = pickle.load(file)

    if not isinstance(models, dict):
        raise TypeError(
            "Expected trained_model.pkl to contain a dictionary of models. "
            "Run the updated training script first."
        )

    results = {
        "test_samples": int(len(y_test)),
        "classes": sorted(y_test.unique().tolist()),
        "models": {},
    }

    report_sections = []
    model_names = list(models.keys())

    figure, axes = plt.subplots(
        1,
        len(model_names),
        figsize=(6 * len(model_names), 5),
        squeeze=False,
    )

    for index, (name, model) in enumerate(models.items()):
        print(f"\nEvaluating {name}...", flush=True)

        predictions = model.predict(X_test)

        accuracy = accuracy_score(y_test, predictions)
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_test,
            predictions,
            average="weighted",
            zero_division=0,
        )

        model_metrics = {
            "accuracy": float(accuracy),
            "weighted_precision": float(precision),
            "weighted_recall": float(recall),
            "weighted_f1": float(f1),
        }

        results["models"][name] = model_metrics

        report = classification_report(
            y_test,
            predictions,
            zero_division=0,
        )

        report_sections.append(
            f"{'=' * 60}\n"
            f"MODEL: {name}\n"
            f"{'=' * 60}\n"
            f"Accuracy: {accuracy:.4f}\n\n"
            f"{report}\n"
        )

        matrix = confusion_matrix(
            y_test,
            predictions,
            labels=results["classes"],
        )

        display = ConfusionMatrixDisplay(
            confusion_matrix=matrix,
            display_labels=results["classes"],
        )
        display.plot(ax=axes[0][index], values_format="d", colorbar=False)
        axes[0][index].set_title(name)

        print(f"Accuracy:           {accuracy:.4f}")
        print(f"Weighted precision: {precision:.4f}")
        print(f"Weighted recall:    {recall:.4f}")
        print(f"Weighted F1-score:  {f1:.4f}")

    figure.tight_layout()
    figure.savefig(CONFUSION_MATRIX_PATH, dpi=200, bbox_inches="tight")
    plt.close(figure)

    with METRICS_PATH.open("w", encoding="utf-8") as file:
        json.dump(results, file, indent=4)

    with REPORT_PATH.open("w", encoding="utf-8") as file:
        file.write("\n".join(report_sections))

    print(f"\nSaved metrics to: {METRICS_PATH}")
    print(f"Saved classification report to: {REPORT_PATH}")
    print(f"Saved confusion matrices to: {CONFUSION_MATRIX_PATH}")


if __name__ == "__main__":
    main()
