# Credit Card Fraud Detection

This project builds and compares machine learning classifiers for detecting fraudulent credit card transactions in a highly imbalanced dataset. It explores how class weighting and classification thresholds affect the balance between detecting fraud and incorrectly flagging legitimate transactions.

## Demo

The Streamlit demo runs the saved Balanced Random Forest with its stored decision threshold. **Single Transaction** lets you analyze predefined examples, compare predictions with actual labels, and inspect the model inputs. The examples include a false positive and missed fraud. **Batch Prediction** accepts a CSV containing all 30 model features, excludes extra columns from inference, and lets you download the original data with fresh `FraudScore` and `Prediction` columns.

From the repository root, install dependencies and launch the app:

```bash
pip install -r requirements.txt
streamlit run app/app.py
```

The app uses `models/credit_card_fraud_model.joblib` and `sample_transactions.csv`. For the existing artifact, it reads the ordered feature names from the fitted model's `feature_names_in_`; packages with a separate `feature_names` entry use that entry instead. Scores are classification scores, not calibrated fraud probabilities. The demo performs inference only, with no training or threshold tuning.

## Project Goal

Fraudulent transactions are extremely rare compared with legitimate transactions. A classifier that predicts every transaction as normal can achieve very high accuracy while detecting no fraud. The project therefore focuses on Precision, Recall, F1 Score, and Average Precision rather than accuracy alone.

## Dataset

The project uses the [Kaggle Credit Card Fraud Detection dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud), collected through a collaboration between Worldline and ULB's Machine Learning Group. It contains European cardholder transactions from September 2013.

| Dataset stage | Transactions | Fraud cases |
|---|---:|---:|
| Original dataset | 284,807 | 492 |
| After duplicate removal | 283,726 | 473 |

Fraud accounts for approximately **0.17%** of the original dataset. The notebook found no missing values and removed 1,081 exact duplicate rows, including 19 fraud transactions, before splitting the data. This treats duplicates as redundant observations and prevents identical rows from appearing across training and evaluation sets.

The columns include:

- `Class`: the target label; `0` means normal and `1` means fraudulent.
- `Amount`: the transaction amount.
- `Time`: seconds elapsed since the first transaction in the dataset.
- `V1`–`V28`: anonymized PCA-transformed features.

The dataset is not included in this repository.

## Exploratory Data Analysis

The notebook examines class imbalance, transaction amounts, relative transaction hours, and feature correlations with the fraud label.

- Fraud transactions have a higher mean amount but a lower median than normal transactions, suggesting a skewed distribution with some larger amounts affecting the mean.
- The relative hour distributions differ between the two classes. `Hour` is derived from `Time` for exploration and excluded from the model inputs.
- `V17`, `V14`, `V12`, and `V10` have the strongest absolute correlations with the fraud label. A closer look at `V17` shows a noticeably lower distribution for fraud transactions.

The hour and `V17` histograms use normalized densities so the two classes can be compared despite their very different sizes. Plots are available within the [notebook](notebooks/Credit_Card_Fraud_Detection.ipynb).

## Data Splitting

The cleaned dataset is split with stratification and `random_state=42` to preserve the rare fraud class in each partition.

| Partition | Share | Transactions | Fraud cases | Purpose |
|---|---:|---:|---:|---|
| Training | 70% | 198,608 | 331 | Model fitting |
| Validation | 15% | 42,559 | 71 | Model comparison and threshold selection |
| Test | 15% | 42,559 | 71 | Final evaluation |

The test set remained untouched during model selection and threshold tuning. Each model uses 30 input features: `Time`, `Amount`, and `V1`–`V28`.

## Baseline

The majority-class baseline predicts every transaction as normal. On the validation set, this gives approximately **99.83% accuracy**, with **Precision = 0** and **Recall = 0**.

Despite its high accuracy, the baseline detects zero fraudulent transactions. This is the clearest example in the project of why accuracy alone is misleading.

## Models Tested

| Model | Configuration |
|---|---|
| Logistic Regression | `StandardScaler` followed by `LogisticRegression` in a pipeline |
| Balanced Logistic Regression | The same pipeline with `class_weight="balanced"` |
| Random Forest | 100 trees, `random_state=42`, `n_jobs=-1` |
| Balanced Random Forest | The same forest configuration with `class_weight="balanced"` |

Scaling for Logistic Regression is fitted on the training set through the pipeline. Random Forest does not require feature scaling. Balanced class weights give the minority fraud class greater importance during training; here, “Balanced Random Forest” refers to scikit-learn's `RandomForestClassifier` with balanced class weights.

## Evaluation Metrics

- **Precision:** how many transactions flagged as fraud are actually fraudulent. Low precision means more legitimate transactions are incorrectly flagged.
- **Recall:** how many actual fraud cases are detected. A false negative is a fraudulent transaction classified as legitimate.
- **F1 Score:** balances Precision and Recall at a chosen threshold.
- **Average Precision (AP):** summarizes the model's precision–recall ranking performance across thresholds, using `average_precision_score`.
- **Confusion Matrix:** shows correct classifications, false positives, and false negatives.

Recall is a priority because missed fraud matters, while Precision remains important to limit false alarms. AP is related to, but is not identical to, the trapezoidal area under a precision–recall curve.

## Threshold Tuning

The default classification threshold of 0.5 is not necessarily optimal for this imbalanced problem. The notebook first demonstrates a manual Logistic Regression threshold of 0.3, then uses `precision_recall_curve` and `np.argmax` to select the F1-maximizing validation threshold for each model.

For the selected Balanced Random Forest, the **F1-maximizing validation threshold was 0.16**. Threshold selection uses validation data only; no thresholds are optimized or adjusted using the test set.

## Model Comparison

**Recorded validation results** from the original experiments:

| Model | Precision | Recall | F1 | Average Precision |
|---|---:|---:|---:|---:|
| Logistic Regression | 0.8030 | 0.7465 | 0.7737 | 0.6877 |
| Balanced Logistic Regression | 0.8871 | 0.7746 | 0.8271 | 0.6971 |
| Random Forest | 0.9825 | 0.7887 | 0.8750 | 0.8222 |
| Balanced Random Forest | 0.9344 | 0.8028 | 0.8636 | 0.8584 |

Regular Random Forest achieved the highest validation Precision and F1. Balanced Random Forest achieved the highest Recall and Average Precision in this comparison, detecting **57 of 71** fraud cases compared with **56 of 71** for regular Random Forest. It also produced more false positives: 4 versus 1. The difference of one detected fraud case is small and does not establish a decisive winner.

Balanced Random Forest was selected because the project prioritizes detecting fraud while maintaining high precision, and it had the strongest ranking performance measured by AP.

The table retains the original recorded results. The cleaned notebook now calculates comparison metrics directly from each model's systematically optimized validation threshold; those edited cells have not been rerun. In particular, the recorded Logistic Regression row came from the earlier manual threshold sweep and should not be read as a newly computed result from the revised tuning code.

## Final Model

**Selected Model:** Balanced Random Forest  
**Classification Threshold:** 0.16

The model and threshold were selected using validation data before the final test evaluation. This choice reflects the project's Precision/Recall trade-off rather than a claim that one model is universally best for fraud detection.

## Final Test Results

The final evaluation used the test set that had been held aside during model and threshold selection. These are the recorded results:

| Metric | Test Result |
|---|---:|
| Precision | 0.8615 |
| Recall | 0.7887 |
| F1 Score | 0.8235 |
| Average Precision | 0.7861 |

Confusion matrix, with actual classes as rows and predicted classes as columns, ordered as normal (`0`), fraud (`1`):

```text
[[42479     9]
 [   15    56]]
```

- 42,479 legitimate transactions correctly classified.
- 9 legitimate transactions incorrectly flagged as fraud.
- 56 fraudulent transactions correctly detected.
- 15 fraudulent transactions missed.

The model detected approximately **79% of fraud cases** in the held-out test set. These results were not used to change the model or its threshold.

## Feature Importance

The notebook reports the following top features for Balanced Random Forest:

| Feature | Importance |
|---|---:|
| V14 | 0.187128 |
| V10 | 0.120271 |
| V4 | 0.099716 |
| V12 | 0.097750 |
| V17 | 0.092815 |
| V3 | 0.080136 |
| V11 | 0.040944 |
| V2 | 0.040890 |
| V16 | 0.039292 |
| V7 | 0.025136 |

Several highly ranked features, including `V14`, `V10`, `V12`, and `V17`, were also prominent in the EDA correlation analysis. These are impurity-based importances: they summarize how useful features were for the model's splits and should not be interpreted as causal evidence.

## Project Workflow

```text
Data Exploration and Missing-Value Analysis
    -> Duplicate Removal
    -> Class Imbalance Analysis and EDA
    -> Stratified Train / Validation / Test Split
    -> Majority-Class Baseline
    -> Logistic Regression and Balanced Logistic Regression
    -> Random Forest and Balanced Random Forest
    -> Validation Threshold Tuning and Precision–Recall Evaluation
    -> Model Comparison and Final Model Selection
    -> Held-Out Test Evaluation
    -> Feature Importance
    -> Joblib Serialization and Artifact Sanity Check
```

## Repository Structure

```text
Credit-Card-Fraud/
├── notebooks/
│   └── Credit_Card_Fraud_Detection.ipynb
├── models/
│   └── credit_card_fraud_model.joblib
├── .gitignore
└── README.md
```

## Running the Project

### Locally

1. Download `creditcard.csv` from the linked Kaggle dataset and place it in `notebooks/` alongside the notebook.
2. Install the libraries used by the project:

   ```bash
   pip install pandas numpy matplotlib scikit-learn joblib
   ```

3. Open `notebooks/Credit_Card_Fraud_Detection.ipynb` in Jupyter or a notebook-capable editor. Set the kernel's working directory to `notebooks/` so the relative CSV path resolves.
4. Run cells in order from a fresh kernel. Use validation data for experiment changes and keep the test set evaluation-only.

### Google Colab

Upload the notebook and `creditcard.csv` to a Colab runtime. Ensure the CSV is in the runtime's working directory, install any missing libraries using the command above, and run cells in order.

The load and save paths are relative to the kernel's working directory. Running the serialization cell writes `credit_card_fraud_model.joblib` there; it does not automatically write to the repository's `models/` directory. Recorded outputs document the original run, and edited experiment outputs have been cleared.

## Saved Model

The trained Balanced Random Forest is serialized with Joblib in [models/credit_card_fraud_model.joblib](models/credit_card_fraud_model.joblib). The existing artifact contains the trained model and selected threshold of **0.16**. Its fitted estimator also retains input feature names in `feature_names_in_`, but the package does not yet contain a separate `feature_names` entry.

The notebook's updated serialization cell saves `model`, `threshold`, and an ordered `feature_names` list for the 30 expected inputs. It then reloads the package and checks probabilities and predictions on five test rows as an artifact sanity check, without tuning or evaluating performance on that sample. Rerunning serialization creates this updated package; the committed artifact is from the original run.

## Key Takeaways

- Approximately 99.8% accuracy can coexist with detecting no fraud at all.
- Precision and Recall expose different errors: false alarms and missed fraud.
- Changing the classification threshold changes the balance between those errors.
- Balanced class weights can improve recall while increasing false positives.
- Random Forest outperformed Logistic Regression on the recorded validation F1 and AP metrics.
- Validation data supports model and threshold selection; the test set is held aside for final evaluation.
