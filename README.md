# Paper Citation Classification

**Course**: UE24CS352A — Machine Learning Mini-Project  
**Authors**: Chinni Krishna Reddy, Saanvi Maharana

1. Project Overview

Paper Citation Classification is a supervised machine-learning project that classifies academic research papers into three citation-count categories: **Low, Medium, and High**.

The system uses a paper's title, abstract, publication year, author count, and primary research field to predict its citation category.

Three models are implemented and evaluated:

- Multinomial Naive Bayes
- Random Forest
- Multilayer Perceptron (MLP) Neural Network

This project classifies recorded citation counts; it does not predict how many citations a paper will receive in the future.

## 2. Problem Statement

Given the title, abstract, publication year, author count, and primary research field of an academic paper, classify it into a Low, Medium, or High citation-count category.

The `citation_count` field is used to create the target label and is excluded from the model input features to prevent target leakage.

## 3. Dataset

The project expects a CSV file at:

`dataset/dataset.csv`

The dataset contains 120,000 records and the following columns:

| Column | Purpose |
|---|---|
| `paper_id` | Paper identifier; not used as a model feature |
| `title` | Paper title |
| `abstract` | Paper abstract |
| `publication_year` | Publication year |
| `author_count` | Number of authors |
| `primary_field` | Primary research field |
| `citation_count` | Used to construct the target category |

The dataset inspection identified 8 missing titles and 2 missing abstracts. The preprocessing pipeline handles missing text and categorical values and validates numeric fields.

### Citation categories

The implemented thresholds are:

| Category | Citation count |
|---|---:|
| Low | 918 or fewer |
| Medium | 919–1,384 |
| High | 1,385 or more |

These thresholds are specific to this project and do not represent universal measures of research quality.

### Training and testing

The data is divided using a stratified 85:15 train/test split with `random_state=42`.

| Split | Records |
|---|---:|
| Training | 102,000 |
| Testing | 18,000 |
| Total | 120,000 |

## 4. Methodology

The implementation follows these steps:

1. Load and validate the dataset.
2. Handle missing and invalid values.
3. Combine paper titles and abstracts into a text feature.
4. Construct citation categories.
5. Split the data into training and testing sets.
6. Transform text and metadata into model features.
7. Train three classification models.
8. Evaluate the models on held-out test data.
9. Save evaluation metrics and confusion-matrix visualizations.

### Text processing

TF-IDF converts text into numerical features. The configured parameters include:

- Maximum features: 20,000
- N-gram range: `(1, 2)`
- English stop-word removal
- Minimum document frequency: 3
- Maximum document frequency: 95%
- Sublinear term-frequency scaling

### Models

**Multinomial Naive Bayes:** A probabilistic text-classification baseline that uses TF-IDF text features.

**Random Forest:** An ensemble of decision trees that uses text features and structured metadata.

**MLP Neural Network:** A feed-forward neural network using TF-IDF features reduced with TruncatedSVD and additional metadata.

The models use different feature representations, so the comparison evaluates the implemented pipelines rather than isolating the algorithms alone.

## 5. Evaluation Metrics

The models are evaluated using:

- **Accuracy:** Fraction of predictions that are correct.
- **Weighted precision:** Precision averaged across classes, weighted by class support.
- **Weighted recall:** Recall averaged across classes, weighted by class support.
- **Weighted F1-score:** Weighted average of class-level F1-scores.
- **Confusion matrix:** Comparison of actual and predicted categories.

## 6. Results

The following results were obtained from the latest recorded local training and evaluation run on 18,000 test records.

| Model | Accuracy | Weighted Precision | Weighted Recall | Weighted F1-score |
|---|---:|---:|---:|---:|
| Multinomial Naive Bayes | 39.39% | 38.72% | 39.39% | 37.99% |
| Random Forest | 38.74% | 39.20% | 38.74% | 38.17% |
| MLP Neural Network | **40.41%** | 39.57% | **40.41%** | **39.42%** |

The MLP Neural Network achieved the highest accuracy and weighted F1-score in this run. These results are specific to the current dataset, preprocessing, model configurations, and train/test split.

### Confusion matrix

The evaluation script generates the combined confusion-matrix figure at:

`results/confusion_matrix.png`

Rows represent actual categories, and columns represent predicted categories. Diagonal entries are correct predictions; off-diagonal entries are misclassifications.

## 7. Installation and Setup

### Prerequisites

- Python 3.11 or a compatible Python version
- The dataset available at `dataset/dataset.csv`

Clone the repository and enter the project directory:

```bash
git clone https://github.com/msaanvi4/Paper-Citation-Classification-.git
cd Paper-Citation-Classification-
```

Create and activate a virtual environment on Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Install the dependencies:

```powershell
python -m pip install -r requirements.txt
```

**Dataset note:** The full dataset is approximately 163 MB. Confirm that the complete CSV is available at the expected path before training. If it is distributed separately, follow the project's documented download instructions.

## 8. Running the Project

Run these commands from the repository root, with the virtual environment activated.

### Train the models

```powershell
python -m src.train
```

This trains the three models and saves the generated model artifact to:

`models/trained_model.pkl`

### Evaluate the models

```powershell
python -m src.evaluate
```

Evaluation outputs are saved in the `results/` directory, including:

- `metrics.json`
- `classification_report.txt`
- `confusion_matrix.png`

### Make a prediction

Example:

```powershell
python -m src.predict --title "Deep Learning for Image Classification" --abstract "This paper studies neural networks for image recognition." --year 2024 --authors 4 --field "Computer Science"
```

The prediction script supports selecting a model with `--model`:

- `Naive Bayes`
- `Random Forest`
- `Neural Network`

The default model is `Neural Network`. The reported confidence is the model's predicted class probability and is not a guarantee of correctness.

### Interactive Demo

After training the models, launch the interactive demo:

```powershell
python demo.py
```

Enter the paper's title, abstract, publication year, author count, and primary research field when prompted. The demo displays predictions from all three trained models.

The demo uses `models/trained_model.pkl`, generated by `python -m src.train`.

## Project Deliverables

- `Paper_Citation.pptx` — project presentation.
- `Paper_Citation_Classification_Report.pdf` — project report.
- `demo.py` — interactive model demonstration.

## 9. Testing

Run the unit tests from the repository root:

```powershell
python -m unittest discover -s tests -v
```

The current test suite covers preprocessing, citation-category boundaries, data splitting, and evaluation-metric validation.

## 10. Repository Structure

```text
Paper-Citation-Classification/
├── dataset/
│   └── dataset.csv
├── models/
│   └── MODEL_GENERATED_BY_TRAINING.txt
├── results/
│   ├── classification_report.txt
│   ├── confusion_matrix.png
│   └── metrics.json
├── src/
│   ├── evaluate.py
│   ├── predict.py
│   ├── preprocessing.py
│   └── train.py
├── tests/
│   ├── test_metrics.py
│   └── test_preprocessing.py
├── .gitattributes
├── .gitignore
├── README.md
└── requirements.txt
```

The trained `.pkl` model is generated locally and is excluded from normal Git tracking by `.gitignore`. The dataset's distribution method must be documented so that another person can reproduce the experiment.

## 11. Limitations

- Citation counts do not directly measure research quality.
- Citation patterns vary across fields and publication years.
- TF-IDF captures word-occurrence patterns but does not fully model semantic meaning.
- The models use a limited set of text and metadata features.
- The models use different feature representations, limiting a direct