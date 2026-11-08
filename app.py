import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from sklearn.metrics import (accuracy_score, roc_auc_score, precision_score,
                             recall_score, f1_score, matthews_corrcoef,
                             ConfusionMatrixDisplay, classification_report)

# Browser tab title, icon and wide layout
st.set_page_config(page_title="Dry Bean Classifier", page_icon="🫘", layout="wide")

st.title("🫘 Dry Bean Classification")
st.write("Upload the test CSV and choose a model in the sidebar to see how it performs.")

# Short explanation of the dataset
with st.expander("About the dataset"):
    st.markdown("""
- **Source:** UCI Machine Learning Repository – Dry Bean Dataset
- **Size:** 13,611 bean grains photographed and measured by a computer vision system
- **Features (16):** size measurements (Area, Perimeter, Major/Minor axis length,
  Convex area, Equivalent diameter, etc.) and shape factors (Roundness,
  Compactness, ShapeFactor1–4, etc.)
- **Classes (7):** BARBUNYA, BOMBAY, CALI, DERMASON, HOROZ, SEKER, SIRA
- **Test set:** 20% of the data (2,723 rows), stratified by class
""")

# Dropdown names mapped to the saved model files
MODEL_FILES = {
    "Logistic Regression": "model/Logistic_Regression.joblib",
    "Decision Tree": "model/Decision_Tree.joblib",
    "KNN": "model/KNN.joblib",
    "Naive Bayes": "model/Naive_Bayes.joblib",
    "SVM": "model/SVM.joblib",
}

# Sidebar controls
st.sidebar.header("⚙️ Settings")
uploaded_file = st.sidebar.file_uploader("Upload test data (CSV)", type="csv")
model_name = st.sidebar.selectbox("Choose a model", list(MODEL_FILES.keys()))

if uploaded_file is None:
    st.info("👈 Upload test_data.csv in the sidebar to get started.")
else:
    data = pd.read_csv(uploaded_file)

    # Data preview hidden in a collapsible box
    with st.expander(f"Preview uploaded data ({data.shape[0]} rows, {data.shape[1]} columns)"):
        st.dataframe(data.head())

    # Separate features and target, load the chosen model, predict
    X_test = data.drop(columns=["Class"])
    y_test = data["Class"]
    model = joblib.load(MODEL_FILES[model_name])
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)

    # Compute the 6 evaluation metrics on the uploaded test data
    scores = {
        "Accuracy": accuracy_score(y_test, y_pred),
        "AUC": roc_auc_score(y_test, y_prob, multi_class="ovr", average="macro"),
        "Precision": precision_score(y_test, y_pred, average="macro"),
        "Recall": recall_score(y_test, y_pred, average="macro"),
        "F1": f1_score(y_test, y_pred, average="macro"),
        "MCC": matthews_corrcoef(y_test, y_pred),
    }

    # Show all 6 metrics in one row
    st.subheader(f"Evaluation metrics – {model_name}")
    cols = st.columns(6)
    for col, (metric, value) in zip(cols, scores.items()):
        col.metric(metric, f"{value:.4f}")

    st.divider()

    # Confusion matrix and classification report in separate tabs
    tab1, tab2 = st.tabs(["📊 Confusion matrix", "📋 Classification report"])

    with tab1:
        plot_col, _ = st.columns([3, 2])
        with plot_col:
            fig, ax = plt.subplots(figsize=(8, 6))
            ConfusionMatrixDisplay.from_predictions(
                y_test, y_pred, ax=ax, cmap="Greens", xticks_rotation=45, colorbar=False)
            ax.set_title(f"{model_name}: actual vs predicted")
            fig.tight_layout()
            st.pyplot(fig)

    with tab2:
        report = classification_report(y_test, y_pred, output_dict=True)
        st.dataframe(pd.DataFrame(report).transpose().round(3))
