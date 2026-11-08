# Dry Bean Classification

## a. Problem Statement
The goal is to predict the type of a dry bean (7 types) using its size and
shape measurements. I trained 5 machine learning models on the same data and
compared them. A Streamlit app is made where you can upload the test data,
pick a model, and see its results.

## b. Dataset Description
- **Source:** UCI Machine Learning Repository – Dry Bean Dataset (id 602)
- **Rows:** 13,611 beans
- **Features:** 16 number columns about size and shape (like Area, Perimeter,
  MajorAxisLength, Roundness, ShapeFactor1–4)
- **Target:** `Class` – 7 bean types: DERMASON (3546), SIRA (2636),
  SEKER (2027), HOROZ (1928), CALI (1630), BARBUNYA (1322), BOMBAY (522)
- **Missing values:** none
- **Split:** 80% for training and 20% for testing (stratified, random_state=42).
  The test part (2,723 rows) is saved as `test_data.csv`.

## c. GitHub Repository Link
https://github.com/Kapila2026aj05655/dry-bean-classification

**Live Streamlit App:** https://dry-bean-classification-skr.streamlit.app

## d. Models Used
1. Logistic Regression
2. Decision Tree
3. K-Nearest Neighbors (k = 5)
4. Naive Bayes (Gaussian)
5. Support Vector Machine (RBF kernel)

Logistic Regression, KNN and SVM use StandardScaler (scaling) in a pipeline.
Since there are 7 classes, Precision, Recall, F1 and AUC are macro averaged
(AUC uses one-vs-rest).

### Comparison Table

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---|---|---|---|---|---|
| Logistic Regression | 0.9207 | 0.9948 | 0.9349 | 0.9314 | 0.9329 | 0.9042 |
| Decision Tree | 0.8917 | 0.9448 | 0.9084 | 0.9088 | 0.9085 | 0.8691 |
| KNN | 0.9166 | 0.9833 | 0.9320 | 0.9271 | 0.9293 | 0.8992 |
| Naive Bayes | 0.7639 | 0.9672 | 0.7744 | 0.7694 | 0.7677 | 0.7154 |
| Support Vector Machine | 0.9221 | 0.9910 | 0.9359 | 0.9330 | 0.9343 | 0.9059 |

### Observations

| ML Model Name | Observation about model performance |
|---|---|
| Logistic Regression | Got 2nd best accuracy (0.9207) and the best AUC (0.9948). It is a simple model but still did very well, so the bean types are not very hard to separate. |
| Decision Tree | Did worse than LR, KNN and SVM (accuracy 0.8917). AUC is also low (0.9448). One tree can overfit the training data. It mixed up SIRA and DERMASON the most (70 SIRA beans predicted as DERMASON). |
| KNN | Very close to the top models (accuracy 0.9166). KNN works on distance, so scaling was needed, otherwise big values like Area would control the result. Made a few more mistakes between BARBUNYA and CALI. |
| Naive Bayes | Lowest scores of all (accuracy 0.7639, MCC 0.7154). It assumes all features are independent, but size features like Area, Perimeter and ConvexArea are related to each other. It got only 46% of BARBUNYA right and predicted 108 of them as CALI. |
| Support Vector Machine | Best accuracy (0.9221), F1 (0.9343) and MCC (0.9059). It can make curved boundaries between classes, so it did a little better than Logistic Regression. |
| **Overall winner** | **SVM** is the winner, with Logistic Regression very close behind. All models got BOMBAY 100% right because it is the biggest bean. SIRA and DERMASON were the most confused pair in every model. |

## Files in this Repository
- `app.py` – Streamlit app (upload CSV, choose model, see metrics,
  confusion matrix and classification report)
- `model_training.ipynb` – loads data, trains the 5 models, checks results,
  saves models (done on BITS Virtual Lab)
- `model/` – saved models (.joblib) for all 5 models + metrics.csv
- `test_data.csv` – test data used in the app
- `requirements.txt` – libraries needed to run the app
