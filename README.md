# Paper Citation Classification

**Course:** UE24CS352A — Machine Learning Mini-Project  
**Authors:** Chinni Krishna Reddy, Saanvi Maharana

## 1. Project Overview

Paper Citation Classification is a supervised machine-learning project that classifies academic research papers into three categories based on their recorded citation counts: **Low, Medium, and High**.

The project uses information about each paper, including its title, abstract, publication year, number of authors, and primary research field, to predict its citation category.

Three machine-learning models are implemented and compared:

- Multinomial Naive Bayes
- Random Forest
- Multilayer Perceptron (MLP) Neural Network

The objective is to evaluate how effectively these models classify papers using textual information and structured metadata.

**Note:** This project classifies citation-count categories from the available dataset. It does not forecast the number of citations a paper will receive in the future.

## 2. Problem Statement

Academic papers receive different numbers of citations depending on several factors, including their research area, publication age, accessibility, and relevance to subsequent research.

The problem addressed in this project is:

> Given the title, abstract, publication year, author count, and primary research field of an academic paper, classify it into a Low, Medium, or High citation-count category.

The `citation_count` column is used to construct the target class. It is excluded from the model input features to prevent target leakage.

### Objectives

- Prepare and inspect an academic-paper dataset.
- Process textual information using TF-IDF.
- Preprocess numerical and categorical metadata.
- Train three machine-learning classifiers.
- Evaluate and compare their performance using multiple metrics.
- Analyse classification errors and identify possible improvements.

## 3. Dataset Description

The project uses the CSV dataset:

`paper_citation_classification_120k.csv`

The dataset contains **120,000 academic-paper records and seven columns**.

| Column | Description | Role |
|---|---|---|
| `paper_id` | Unique paper identifier or OpenAlex work URL | Identifier |
| `title` | Title of the research paper | Text feature |
| `abstract` | Summary of the research paper | Text feature |
| `publication_year` | Year of publication | Numerical feature |
| `author_count` | Number of authors | Numerical feature |
| `primary_field` | Primary research field | Categorical feature |
| `citation_count` | Recorded number of citations | Used to construct the target |

During dataset inspection, 8 missing titles and 2 missing abstracts were identified.

The model uses the title, abstract, publication year, author count, and primary research field as inputs. The paper identifier is not used as a model feature.

### 3.1 Target Class Construction

The project documentation records these thresholds:

| Class | Citation count |
|---|---:|
| Low | 918 or fewer |
| Medium | 919–1,384 |
| High | 1,385 or more |

These thresholds are dataset-specific and should not be interpreted as universal definitions of research quality or impact.

**Before submission, verify that these boundaries match the label-creation cell in the final notebook.** The README, notebook, and saved results must use the same thresholds.

### 3.2 Training and Testing Split

The dataset is divided into training and testing sets using an 85:15 split.

| Dataset | Number of records |
|---|---:|
| Training set | 102,000 |
| Testing set | 18,000 |
| Total | 120,000 |

The split uses `random_state=42` for reproducibility.

The training set is used to fit the models, while the testing set is reserved for evaluating their performance on held-out records.

## 4. Methodology

The project follows these main steps:

1. Load and inspect the dataset.
2. Check missing values and prepare the text fields.
3. Create Low, Medium, and High target labels.
4. Split the records into training and testing sets.
5. Convert textual information into numerical features.
6. Process numerical and categorical metadata.
7. Train the three classification models.
8. Evaluate model predictions and compare the results.
9. Generate confusion matrices and save the evaluation outputs.

### 4.1 Text Preprocessing with TF-IDF

Machine-learning models require numerical inputs. The title and abstract are combined into a text field and transformed using **Term Frequency–Inverse Document Frequency (TF-IDF)**.

TF-IDF assigns weights to words based on their occurrence in a document and their distribution across the document collection.

The project documentation records these TF-IDF settings:

- Maximum features: 20,000
- N-gram range: `(1, 2)`
- English stop-word removal
- Minimum document frequency: 3
- Maximum document frequency: 95%
- Sublinear term-frequency scaling

Unigrams represent individual words, while bigrams represent pairs of consecutive words.

### 4.2 Metadata Preprocessing

The structured metadata is processed using the following techniques:

- **StandardScaler:** Standardizes numerical features such as publication year and author count.
- **OneHotEncoder:** Converts primary research-field categories into numerical binary features.
- **ColumnTransformer:** Applies the appropriate transformation to each feature type.

Preprocessing parameters are learned from the training data and then applied to the testing data. This prevents the test set from influencing the fitted preprocessing transformations.

### 4.3 Machine-Learning Models

#### A. Multinomial Naive Bayes

Multinomial Naive Bayes is a probabilistic classifier commonly used for text classification. It estimates the likely class based on the observed text features.

In this project, it serves as a baseline model using TF-IDF text features.

#### B. Random Forest

Random Forest combines multiple decision trees to generate a final prediction. It can learn nonlinear relationships and interactions between features.

The documented configuration includes 100 trees, a maximum depth of 20, a minimum leaf size of 5, and `random_state=42`.

The model uses text features and structured metadata.

#### C. MLP Neural Network

A Multilayer Perceptron is a feed-forward neural network that learns patterns through interconnected layers of computational units.

The documented configuration uses one hidden layer of 64 ReLU units, the Adam optimizer, a learning rate of 0.001, a batch size of 256, and a maximum of 15 iterations with early stopping.

TruncatedSVD reduces the combined feature matrix to 200 components before the neural network is trained.

**Note:** The three models use different feature representations. Their results compare the implemented pipelines rather than isolating the algorithm as the only difference.

## 5. Evaluation Metrics

The models are evaluated using the following metrics:

- **Accuracy:** The proportion of all predictions that are correct.
- **Precision:** How often predictions for a class are correct.
- **Recall:** The proportion of actual members of a class correctly identified.
- **F1-score:** A measure combining precision and recall.
- **Confusion matrix:** A table comparing actual and predicted classes.

Weighted precision, recall, and F1-score account for the number of test examples belonging to each class.

## 6. Results

The project report records the following results on the 18,000-record test set.

| Model | Accuracy | Weighted Precision | Weighted Recall | Weighted F1-score |
|---|---:|---:|---:|---:|
| Multinomial Naive Bayes | 39.36% | 38.53% | 39.36% | 37.81% |
| Random Forest | 39.68% | 39.80% | 39.68% | **39.36%** |
| MLP Neural Network | **40.70%** | 39.89% | 40.70% | 38.93% |

### Key Findings

- The MLP Neural Network achieved the highest accuracy of **40.70%**.
- Random Forest achieved the highest weighted F1-score of **0.3936**.
- All three models produced classification errors, as shown by their confusion matrices.
- The results indicate that citation classification remains challenging using the selected features.

### Confusion Matrices

The notebook generates confusion matrices for the three models.

The rows represent the actual citation categories, and the columns represent the predicted categories. Diagonal cells indicate correct predictions, while off-diagonal cells indicate misclassifications.

If the generated image is committed at `Results/confusion_matrices.png`, it can be displayed in this section:

![Confusion matrices for the three classification models](Results/confusion_matrices.png)

The image path must match the actual location and capitalization of the file in the repository.

## 7. How to Run the Project

### Option A: Google Colab

1. Open Google Colab.
2. Upload the project notebook.
3. Connect to a runtime.
4. Run the notebook cells in order.
5. When prompted by the upload cell, upload `paper_citation_classification_120k.csv`.
6. Continue through data preprocessing, model training, evaluation, and result generation.
7. Verify that the evaluation outputs are produced and save the files needed for submission.

Colab runtime files are temporary. Download or copy any outputs that need to be retained.

### Option B: Local Jupyter Notebook

**Prerequisites:** Python 3.11 or a compatible Python version and the required packages.

Install the dependencies:

```bash
pip install numpy pandas scikit-learn matplotlib joblib jupyter
```

Start Jupyter:

```bash
jupyter notebook
```

Open the project notebook and run its cells in order.

Ensure that the CSV is stored at the path expected by the notebook. If the dataset is stored in a `data/` folder, the data-loading statement should use:

```python
df = pd.read_csv("data/paper_citation_classification_120k.csv")
```

Use this path only when the CSV actually exists in that location.

## 8. Repository Structure

The following is an example of the intended repository layout. Keep only entries that are actually present in the submitted repository.

```text
Paper-Citation-Classification/
├── data/
│   └── paper_citation_classification_120k.csv
├── Results/
│   ├── Results.md
│   ├── model_comparison.csv
│   └── confusion_matrices.png
├── model/
│   └── trained model files, if saved
├── 02_Citation_Classification.ipynb
├── Paper_Citation.pptx
└── README.md
```

The notebook filename, folder names, and output filenames in this tree must match the actual repository.

## 9. Limitations

- Citation counts are an imperfect measure of scholarly attention and do not directly measure research quality.
- Citation patterns vary across research fields and publication ages.
- TF-IDF captures word-occurrence patterns but does not fully understand semantic meaning or context.
- The project uses a limited set of metadata features.
- The model pipelines use different feature representations, limiting a direct algorithm-only comparison.
- The evaluation uses one train/test split. Additional validation and hyperparameter tuning would help assess robustness.
- The documented MLP configuration uses a maximum of 15 iterations; further tuning may improve its performance.

## 10. Future Work

Potential improvements include:

- Tune model hyperparameters using a validation set.
- Compare models using a consistent feature-processing pipeline.
- Explore additional reliable metadata, such as publication venue and paper age.
- Experiment with scientific-text embeddings such as SciBERT.
- Evaluate alternative citation thresholds.
- Assess the stability of results across multiple splits.

## 11. References

- OpenAlex: https://openalex.org
- Scikit-learn documentation: https://scikit-learn.org/stable/
- Bui, C., Saavedra, C., and Liu, S. (2019). *Paper Citation Classification*. Stanford CS229 Machine Learning Final Report.

## 12. Reproducibility Checklist

Before submitting the project, verify that:

- The notebook and README use the same citation thresholds.
- The reported metrics match the final executed notebook.
- The dataset source URL, version, and applicable license are documented.
- The notebook's filename and all referenced output paths are correct.
- The required dataset, notebook, and result files are available to the intended reviewers.
- No credentials or private information are included in the repository.
