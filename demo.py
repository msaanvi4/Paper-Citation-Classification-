"""Interactive demo for Paper Citation Classification.

Run from the project root:
    python demo.py

Prerequisite:
    Train the models using: python -m src.train

The demo uses models/trained_model.pkl, which contains all three
trained scikit-learn pipelines.
"""

from src.predict import AVAILABLE_MODELS, predict_paper


def ask_nonempty(prompt):
    """Ask for a non-empty text value."""
    while True:
        value = input(prompt).strip()
        if value:
            return value
        print("Please enter a value.")


def ask_number(prompt, value_type, minimum=None, maximum=None):
    """Read and validate a numeric input."""
    while True:
        try:
            value = value_type(input(prompt).strip())

            if minimum is not None and value < minimum:
                print(f"Value must be at least {minimum}.")
                continue

            if maximum is not None and value > maximum:
                print(f"Value must be at most {maximum}.")
                continue

            return value

        except ValueError:
            print("Please enter a valid number.")


def run_demo():
    """Collect paper details and display predictions from all models."""
    print("=" * 58)
    print("       PAPER CITATION CLASSIFICATION - DEMO")
    print("=" * 58)
    print("Predict whether a research paper belongs to Low, Medium,")
    print("or High citation category.\n")

    while True:
        title = ask_nonempty("Paper title: ")
        abstract = ask_nonempty("Paper abstract: ")

        publication_year = ask_number(
            "Publication year (e.g. 2023): ",
            int,
            minimum=1900,
            maximum=2100,
        )

        author_count = ask_number(
            "Number of authors: ",
            int,
            minimum=1,
        )

        primary_field = ask_nonempty(
            "Primary field (e.g. Computer Science): "
        )

        print("\nPREDICTIONS")
        print("-" * 58)

        for model_name in AVAILABLE_MODELS:
            try:
                result = predict_paper(
                    title=title,
                    abstract=abstract,
                    publication_year=publication_year,
                    author_count=author_count,
                    primary_field=primary_field,
                    model_name=model_name,
                )

                print(
                    f"{model_name:<18}: "
                    f"{result['citation_category']}"
                )

                if result["confidence"] is not None:
                    print(
                        f"{'':18}  "
                        f"Confidence: {result['confidence']:.1%} "
                        "(not a guarantee)"
                    )

            except FileNotFoundError:
                print(
                    "\nTrained models are missing. Run "
                    "'python -m src.train' first."
                )
                return

            except Exception as exc:
                print(f"{model_name} prediction failed: {exc}")

        print("-" * 58)
        print(
            "Note: predictions classify recorded citation categories; "
            "they do not guarantee future citation counts."
        )

        again = input("\nPredict another paper? (y/n): ").strip().lower()
        if again not in {"y", "yes"}:
            print("\nDemo finished.")
            break

        print()


if __name__ == "__main__":
    run_demo()