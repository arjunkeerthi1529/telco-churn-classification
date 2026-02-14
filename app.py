import streamlit as st
import pandas as pd
import joblib
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    matthews_corrcoef,
    confusion_matrix
)

# -------------------------------
# Page Configuration
# -------------------------------
st.set_page_config(
    page_title="Telco Churn ML App",
    layout="wide"
)

# -------------------------------
# Custom CSS Styling
# -------------------------------
st.markdown("""
    <style>
        .main {
            background-color: #f5f7fa;
        }
        .title-style {
            font-size: 36px;
            font-weight: bold;
            color: #1f4e79;
        }
        .section-header {
            font-size: 24px;
            font-weight: bold;
            color: #0e6ba8;
            margin-top: 20px;
        }
        .metric-container {
            background-color: white;
            padding: 15px;
            border-radius: 10px;
            box-shadow: 2px 2px 8px rgba(0,0,0,0.1);
        }
    </style>
""", unsafe_allow_html=True)

# -------------------------------
# Title
# -------------------------------
st.markdown('<p class="title-style">📊 Telco Customer Churn Prediction App</p>', unsafe_allow_html=True)
st.write("Upload RAW test dataset (original columns) to evaluate trained ML models.")

# -------------------------------
# Load Models & Utilities
# -------------------------------
log_model = joblib.load("model/logistic_model.pkl")
dt_model = joblib.load("model/decision_tree_model.pkl")
knn_model = joblib.load("model/knn_model.pkl")
nb_model = joblib.load("model/naive_bayes_model.pkl")
rf_model = joblib.load("model/random_forest_model.pkl")
xgb_model = joblib.load("model/xgboost_model.pkl")

scaler = joblib.load("model/scaler.pkl")
feature_columns = joblib.load("model/feature_columns.pkl")

# -------------------------------
# Model Selection
# -------------------------------
st.markdown('<p class="section-header">🔍 Select Model</p>', unsafe_allow_html=True)

model_name = st.selectbox(
    "",
    [
        "Logistic Regression",
        "Decision Tree",
        "KNN",
        "Naive Bayes",
        "Random Forest",
        "XGBoost"
    ]
)

# -------------------------------
# File Upload
# -------------------------------
st.markdown('<p class="section-header">📁 Upload Test Dataset</p>', unsafe_allow_html=True)

uploaded_file = st.file_uploader("", type=["csv"])

if uploaded_file is not None:

    data = pd.read_csv(uploaded_file)

    st.markdown('<p class="section-header">📄 Data Preview</p>', unsafe_allow_html=True)
    st.dataframe(data.head())

    if "Churn" not in data.columns:
        st.error("Uploaded file must contain 'Churn' column.")
        st.stop()

    # Target Handling
    if data["Churn"].dtype == "object":
        y_test = data["Churn"].map({"Yes": 1, "No": 0})
    else:
        y_test = pd.to_numeric(data["Churn"], errors="coerce")

    valid_idx = y_test.notna()
    y_test = y_test[valid_idx].astype(int)
    X_test = data.drop("Churn", axis=1)
    X_test = X_test[valid_idx]

    if len(y_test) == 0:
        st.error("No valid target values found.")
        st.stop()

    # Encoding
    X_test = pd.get_dummies(X_test, drop_first=True)
    X_test = X_test.reindex(columns=feature_columns, fill_value=0)

    # Scaling if required
    if model_name in ["Logistic Regression", "KNN", "Naive Bayes"]:
        X_test = scaler.transform(X_test)

    # Select Model
    model_dict = {
        "Logistic Regression": log_model,
        "Decision Tree": dt_model,
        "KNN": knn_model,
        "Naive Bayes": nb_model,
        "Random Forest": rf_model,
        "XGBoost": xgb_model
    }

    model = model_dict[model_name]

    # Predictions
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    # Metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob)
    mcc = matthews_corrcoef(y_test, y_pred)

    # -------------------------------
    # Display Metrics
    # -------------------------------
    st.markdown('<p class="section-header">📈 Model Performance</p>', unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)
    col1.metric("Accuracy", f"{accuracy:.4f}")
    col2.metric("Precision", f"{precision:.4f}")
    col3.metric("Recall", f"{recall:.4f}")

    col4, col5, col6 = st.columns(3)
    col4.metric("F1 Score", f"{f1:.4f}")
    col5.metric("AUC Score", f"{auc:.4f}")
    col6.metric("MCC Score", f"{mcc:.4f}")

    # -------------------------------
    # Confusion Matrix
    # -------------------------------
    st.markdown('<p class="section-header">🔢 Confusion Matrix</p>', unsafe_allow_html=True)
    st.write(confusion_matrix(y_test, y_pred))
