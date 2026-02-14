# 📊 Telco Customer Churn Classification

## 📌 1. Problem Statement

Customer churn is a major challenge in the telecom industry. When customers discontinue services, companies lose recurring revenue and must spend additional resources to acquire new customers.

The objective of this project is to build, evaluate, and compare multiple machine learning classification models to predict whether a telecom customer will churn based on demographic, service usage, and billing-related features.

This project also includes deployment of the trained models using a Streamlit web application.

---

## 📂 2. Dataset Description

**Dataset Used:** Telco Customer Churn Dataset  
**Total Instances (after cleaning):** 7032  
**Original Features:** 20  
**Features After Encoding:** 30  

### Important Features Include:

- Gender  
- SeniorCitizen  
- Partner  
- Dependents  
- Tenure  
- PhoneService  
- InternetService  
- OnlineSecurity  
- Contract  
- MonthlyCharges  
- TotalCharges  
- PaymentMethod  

### Target Variable:

**Churn**
- 1 → Customer Churned  
- 0 → Customer Stayed  

---

## 🧹 3. Data Preprocessing

The following preprocessing steps were performed:

- Removed 11 rows with missing `TotalCharges`
- Converted `TotalCharges` to numeric
- One-hot encoding for categorical features
- Train-test split (80% training, 20% testing)
- Feature scaling using StandardScaler (for Logistic Regression, KNN, and Naive Bayes)

---

## 🤖 4. Machine Learning Models Implemented

Six classification models were trained and evaluated:

1. Logistic Regression  
2. Decision Tree  
3. K-Nearest Neighbors (KNN)  
4. Naive Bayes  
5. Random Forest  
6. XGBoost  

All models were trained on the same dataset split for fair comparison.

---

## 📏 5. Evaluation Metrics

Each model was evaluated using the following metrics:

- Accuracy  
- Precision  
- Recall  
- F1 Score  
- AUC (Area Under ROC Curve)  
- MCC (Matthews Correlation Coefficient)  

These metrics help measure both prediction performance and handling of class imbalance.

---

## 📊 6. Model Performance Comparison

| Model | Accuracy | AUC | Precision | Recall | F1 Score | MCC |
|--------|----------|------|------------|--------|----------|------|
| Logistic Regression | 0.8038 | 0.8357 | 0.6476 | 0.5749 | 0.6091 | 0.4803 |
| Decision Tree | 0.7093 | 0.6403 | 0.4543 | 0.4652 | 0.4597 | 0.2609 |
| KNN | 0.7527 | 0.7661 | 0.5357 | 0.5214 | 0.5285 | 0.3609 |
| Naive Bayes | 0.6446 | 0.8102 | 0.4184 | 0.8636 | 0.5637 | 0.3808 |
| Random Forest | 0.7896 | 0.8176 | 0.6364 | 0.4866 | 0.5515 | 0.4237 |
| XGBoost | 0.7790 | 0.8348 | 0.5994 | 0.5080 | 0.5499 | 0.4072 |

---

## 🔎 7. Observations

- Logistic Regression achieved the highest overall balanced performance and the best MCC score.
- XGBoost achieved a high AUC score and strong probability ranking.
- Random Forest significantly improved over a single Decision Tree due to ensemble learning.
- Naive Bayes achieved high recall, making it useful when detecting churn cases is more important than precision.
- Decision Tree showed weaker generalization compared to ensemble methods.

Overall, Logistic Regression provided the best balance of accuracy, AUC, and MCC for this dataset.

---

## 🚀 8. Streamlit Web Application

A Streamlit application was developed with the following features:

- CSV Dataset Upload Option  
- Model Selection Dropdown  
- Real-time Evaluation Metrics  
- Confusion Matrix Display  
- Clean and Styled User Interface  

### To Run Locally:
- streamlit run app.py


---

## 📦 9. Project Structure

telco-churn-classification/
│-- app.py
│-- requirements.txt
│-- README.md
│-- model/
│-- logistic_model.pkl
│-- decision_tree_model.pkl
│-- knn_model.pkl
│-- naive_bayes_model.pkl
│-- random_forest_model.pkl
│-- xgboost_model.pkl
│-- scaler.pkl
│-- feature_columns.pkl


---

## 🛠 10. Technologies Used

- Python  
- Scikit-learn  
- XGBoost  
- Pandas  
- NumPy  
- Streamlit  

---

## 🌐 11. Deployment

The project is deployed using Streamlit Cloud and connected to a public GitHub repository.  
Users can upload test data, select a model, and view real-time performance metrics.

---

## ✅ 12. Conclusion

This project demonstrates:

- Implementation of multiple classification algorithms  
- Comparative performance analysis  
- Use of six evaluation metrics  
- Model deployment using Streamlit  

Logistic Regression provided the most balanced and stable performance for churn prediction in this dataset.
