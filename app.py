import streamlit as st
import pandas as pd
import joblib
from sklearn.metrics import (accuracy_score, roc_auc_score, precision_score,
                             recall_score, f1_score, matthews_corrcoef)
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay, classification_report

# Page title and short description
st.title("Dry Bean Classification")
st.write("Upload the test CSV, choose a model, and see how it performs.")

# Dropdown names mapped to the saved model files
MODEL_FILES = {
    "Logistic Regression": "model/Logistic_Regression.joblib",
    "Decision Tree": "model/Decision_Tree.joblib",
    "KNN": "model/KNN.joblib",
    "Naive Bayes": "model/Naive_Bayes.joblib",
    "SVM": "model/SVM.joblib",
}

# CSV upload and model selection
uploaded_file = st.file_uploader("Upload test data (CSV)", type="csv")
model_name = st.selectbox("Choose a model", list(MODEL_FILES.keys()))

if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
    st.write("Rows:", data.shape[0], " Columns:", data.shape[1])
    st.dataframe(data.head())

    # Separate features and target, load the chosen model, predict
    X_test = data.drop(columns=["Class"])
    y_test = data["Class"]
    model = joblib.load(MODEL_FILES[model_name])
    y_pred = model.predict(X_test)
        # Compute the 6 evaluation metrics on the uploaded test data
    y_prob = model.predict_proba(X_test)
    scores = {
        "Accuracy": accuracy_score(y_test, y_pred),
        "AUC": roc_auc_score(y_test, y_prob, multi_class="ovr", average="macro"),
        "Precision": precision_score(y_test, y_pred, average="macro"),
        "Recall": recall_score(y_test, y_pred, average="macro"),
        "F1": f1_score(y_test, y_pred, average="macro"),
        "MCC": matthews_corrcoef(y_test, y_pred),
    }

    # Show metrics as cards, 3 per row
    st.subheader("Evaluation metrics")
    cols = st.columns(3)
    for i, (metric, value) in enumerate(scores.items()):
        cols[i % 3].metric(metric, f"{value:.4f}")
    st.success(f"{model_name} made predictions for {len(y_pred)} rows")
        # Confusion matrix: rows = actual class, columns = predicted class
    st.subheader("Confusion matrix")
    fig, ax = plt.subplots(figsize=(7, 6))
    ConfusionMatrixDisplay.from_predictions(
        y_test, y_pred, ax=ax, cmap="Greens", xticks_rotation=45, colorbar=False)
    st.pyplot(fig)

    # Per-class precision, recall and F1
    st.subheader("Classification report")
    report = classification_report(y_test, y_pred, output_dict=True)
    st.dataframe(pd.DataFrame(report).transpose().round(4))