# Credit Default Risk Classification - ML Assignment 2

## 1. Problem Statement

Banks have to decide whether a loan applicant is likely to repay or to default. A wrong
approval costs the lender money, while a wrong rejection loses a good customer. The goal of
this project is to build and compare five classical classification models that predict, from
the applicant's profile and loan details, whether the applicant will **default (1)** or
**repay (0)**, and to present the trained models in an interactive Streamlit web app.

## 2. Dataset Description  [1 mark]

| Item | Details |
| --- | --- |
| Name | Statlog (German Credit Data) |
| Source | UCI Machine Learning Repository |
| Type | Binary classification |
| Instances | 1000 |
| Input features | 20 (7 numeric, 13 categorical) |
| Target | `default` - 1 = applicant defaulted (bad credit), 0 = repaid (original file codes 1/2, recoded to 0/1) |
| Class balance | 700 repaid / 300 defaulted (70 % / 30 %) |
| Missing values | None |

Feature groups: checking and savings balance, credit history, loan purpose, loan amount and
duration, installment rate, employment length, personal status, other debtors, property,
age, other installment plans, housing, number of existing credits, job type, dependents,
telephone and foreign-worker flag.

**Pre-processing** (stored inside every saved model pipeline, so raw CSV rows can be scored directly):
numeric columns are standardised with `StandardScaler`; categorical columns are one-hot encoded
with `OneHotEncoder(handle_unknown="ignore")`.

**Split:** stratified 80 / 20 (800 train, 200 test, `random_state=2026`). The 200 test rows are
saved as `test_data.csv`. Hyper-parameters of every model were picked by 5-fold cross-validated
grid search on the training part only (scoring = ROC-AUC).

**GitHub Repository Link:** https://github.com/kaushiknb98/ml-project-assignment

**Live Streamlit App:** https://loan-risk-desk.streamlit.app/

### Repository layout

```
project-folder/
|-- app.py                  
|-- train_models.py       
|-- requirements.txt
|-- README.md
|-- test_data.csv          
|-- data/german_credit.csv  
|-- model/
|   |-- logistic_regression.joblib
|   |-- decision_tree.joblib
|   |-- knn.joblib
|   |-- naive_bayes.joblib
|   |-- svm.joblib
|   |-- metrics_summary.csv
|   `-- best_params.json
```

## 3. Models Used  [5 marks]

All five models were trained on the same training split and evaluated on the same 200-row test set.
Precision, Recall, F1 are reported for the positive class (default = 1).

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
| --- | --- | --- | --- | --- | --- | --- |
| Logistic Regression | 0.7850 | 0.8154 | 0.6735 | 0.5500 | 0.6055 | 0.4643 |
| Decision Tree | 0.6950 | 0.6992 | 0.4909 | 0.4500 | 0.4696 | 0.2566 |
| KNN | 0.7450 | 0.7893 | 0.7368 | 0.2333 | 0.3544 | 0.3089 |
| Naive Bayes | 0.7350 | 0.7805 | 0.5522 | 0.6167 | 0.5827 | 0.3907 |
| Support Vector Machine | 0.7650 | 0.8164 | 0.6512 | 0.4667 | 0.5437 | 0.4010 |

Selected hyper-parameters: Logistic Regression C = 0.1; Decision Tree max_depth = 7,
min_samples_leaf = 10; KNN k = 21 with distance weighting; Gaussian Naive Bayes
var_smoothing = 0.1; SVM (RBF kernel) C = 10, gamma = 0.01.

### Observations on model performance  [3 marks]

| ML Model Name | Observation about model performance |
| --- | --- |
| Logistic Regression | Best accuracy (78.5 %), best F1 (0.606) and best MCC (0.464), with an AUC almost equal to the top score. The one-hot encoded features are largely additive in their effect on default risk, so a regularised linear model (C = 0.1) fits well and does not over-fit the 800 training rows. |
| Decision Tree | Weakest model on every metric (accuracy 69.5 %, AUC 0.70, MCC 0.26). Accuracy is close to the 70 % that a model predicting "repaid" for everybody would reach. A single tree is unstable on a small dataset with many sparse dummy columns and over-fits even after depth and leaf-size limits. |
| KNN | Highest precision (0.737) but by far the lowest recall (0.233): it flags very few applicants as defaulters, and most real defaulters slip through. Distance in a space of 61 encoded columns is not very informative, and with k = 21 the majority class dominates the neighbourhood. Not suitable when missing a defaulter is costly. |
| Naive Bayes | Highest recall (0.617), so it catches the most defaulters, but at the cost of lower precision (0.552) and accuracy (73.5 %). The independence assumption is violated by correlated features such as loan amount and duration, which makes its probabilities less reliable, yet it is a useful choice when recall matters most. |
| Support Vector Machine | Best AUC (0.816) and second-best accuracy (76.5 %) and MCC (0.401), so its ranking of risky versus safe applicants is excellent. At the default 0.5 threshold, though, recall is only 0.467. Lowering the decision threshold would likely improve recall. |
| Overall, Winner for your dataset? | **Logistic Regression.** It leads on accuracy, F1 and MCC, and its AUC (0.8154) is within 0.001 of the SVM. It is also the simplest, fastest and most interpretable model. If the bank cares most about catching defaulters, Naive Bayes (recall 0.617) is the alternative. |

## 4. Streamlit App Features

- CSV upload of test data (features + `default` column); a sample file can be downloaded from the sidebar
- Model selection dropdown with all five classifiers
- Evaluation metrics (Accuracy, AUC, Precision, Recall, F1, MCC) for the selected model
- Confusion matrix and classification report (plus an optional ROC curve)
- Tab comparing all five models on the same uploaded data
- Tab listing predictions with default probability, a filter for misclassified rows and a download button

The app loads the saved `.joblib` pipelines and only calls `predict` / `predict_proba`; nothing is re-trained.

## 5. How to Run Locally

```bash
pip install -r requirements.txt
python train_models.py      # optional - re-creates model/ and test_data.csv
streamlit run app.py
```
