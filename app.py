import streamlit as st
import pandas as pd
import joblib
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    matthews_corrcoef,
    confusion_matrix,
    roc_curve
)

# ---------------------------------------------------------
# PAGE CONFIGURATION
# ---------------------------------------------------------
st.set_page_config(
    page_title="Telco Churn Prediction Dashboard",
    page_icon="📊",
    layout="wide"
)

st.title("📊 Telco Customer Churn Prediction Dashboard")
st.markdown("Compare multiple machine learning models interactively.")

# ---------------------------------------------------------
# LOAD MODELS (CACHED)
# ---------------------------------------------------------
@st.cache_resource
def load_models():
    models = {
        "Logistic Regression": joblib.load("model/logistic_model.pkl"),
        "Decision Tree": joblib.load("model/decision_tree_model.pkl"),
        "KNN": joblib.load("model/knn_model.pkl"),
        "Naive Bayes": joblib.load("model/naive_bayes_model.pkl"),
        "Random Forest": joblib.load("model/random_forest_model.pkl"),
        "XGBoost": joblib.load("model/xgboost_model.pkl"),
    }
    scaler = joblib.load("model/scaler.pkl")
    feature_columns = joblib.load("model/feature_columns.pkl")
    return models, scaler, feature_columns


models, scaler, feature_columns = load_models()

# ---------------------------------------------------------
# SIDEBAR CONTROLS
# ---------------------------------------------------------
st.sidebar.header("⚙️ Controls")

model_name = st.sidebar.selectbox(
    "Select Model",
    list(models.keys())
)

uploaded_file = st.sidebar.file_uploader(
    "Upload Test Dataset",
    type=["csv", "xlsx", "xls", "json"]
)

# ---------------------------------------------------------
# FILE HANDLING
# ---------------------------------------------------------
def load_data(file):
    file_type = file.name.split(".")[-1]

    if file_type == "csv":
        return pd.read_csv(file)
    elif file_type in ["xlsx", "xls"]:
        return pd.read_excel(file)
    elif file_type == "json":
        return pd.read_json(file)
    else:
        return None


if uploaded_file is not None:

    data = load_data(uploaded_file)

    if data is None:
        st.error("Unsupported file format.")
        st.stop()

    st.subheader("📄 Data Preview")
    st.dataframe(data.head(), use_container_width=True)

    if "Churn" not in data.columns:
        st.error("Uploaded file must contain a 'Churn' column.")
        st.stop()

    # ---------------------------------------------------------
    # TARGET PROCESSING
    # ---------------------------------------------------------
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

    # ---------------------------------------------------------
    # ENCODING
    # ---------------------------------------------------------
    X_test = pd.get_dummies(X_test, drop_first=True)
    X_test = X_test.reindex(columns=feature_columns, fill_value=0)

    # ---------------------------------------------------------
    # SCALING
    # ---------------------------------------------------------
    if model_name in ["Logistic Regression", "KNN", "Naive Bayes"]:
        X_test = scaler.transform(X_test)

    model = models[model_name]

    # ---------------------------------------------------------
    # PREDICTIONS
    # ---------------------------------------------------------
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    # ---------------------------------------------------------
    # METRICS
    # ---------------------------------------------------------
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob)
    mcc = matthews_corrcoef(y_test, y_pred)

    st.subheader("📈 Model Performance")

    # -------- Styled Metric Function --------
    def styled_metric(label, value, highlight=False):
        color = "#1f77b4" if highlight else "#333"
        bg = "#e8f2ff" if highlight else "#f4f4f4"

        st.markdown(
            f"""
            <div style="
                background-color:{bg};
                padding:20px;
                border-radius:12px;
                text-align:center;
                margin-bottom:15px;
            ">
                <div style="font-size:18px; font-weight:600; color:{color};">
                    {label}
                </div>
                <div style="font-size:32px; font-weight:bold; color:{color};">
                    {value:.4f}
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    col1, col2, col3 = st.columns(3)
    col4, col5, col6 = st.columns(3)

    with col1:
        styled_metric("Accuracy", accuracy, highlight=True)

    with col2:
        styled_metric("Precision", precision)

    with col3:
        styled_metric("Recall", recall)

    with col4:
        styled_metric("F1 Score", f1)

    with col5:
        styled_metric("AUC Score", auc, highlight=True)

    with col6:
        styled_metric("MCC Score", mcc)

    # ---------------------------------------------------------
    # ROC CURVE
    # ---------------------------------------------------------
    st.subheader("📉 ROC Curve")

    fpr, tpr, _ = roc_curve(y_test, y_prob)

    fig1, ax1 = plt.subplots()
    ax1.plot(fpr, tpr)
    ax1.plot([0, 1], [0, 1])
    ax1.set_xlabel("False Positive Rate")
    ax1.set_ylabel("True Positive Rate")
    ax1.set_title("ROC Curve")
    st.pyplot(fig1)

    # ---------------------------------------------------------
    # CONFUSION MATRIX
    # ---------------------------------------------------------
    st.subheader("🔢 Confusion Matrix")

    cm = confusion_matrix(y_test, y_pred)

    fig2, ax2 = plt.subplots()
    sns.heatmap(cm, annot=True, fmt="d", ax=ax2)
    ax2.set_xlabel("Predicted")
    ax2.set_ylabel("Actual")
    st.pyplot(fig2)

    # ---------------------------------------------------------
    # METRIC EXPLANATION
    # ---------------------------------------------------------
    with st.expander("ℹ️ What do these metrics mean?"):
        st.write("""
        - **Accuracy**: Overall correctness of the model  
        - **Precision**: Percentage of predicted churn cases that were correct  
        - **Recall**: Percentage of actual churn cases detected  
        - **F1 Score**: Balance between precision and recall  
        - **AUC**: Model's ability to distinguish between classes  
        - **MCC**: Balanced measure even for imbalanced datasets  
        """)

else:
    st.info("Upload a dataset from the sidebar to begin.")
