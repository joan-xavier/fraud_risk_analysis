
# Credit Card Fraud Detection Dashboard

An interactive Streamlit dashboard that analyzes and predicts fraudulent credit card transactions using machine learning. This app uses logistic regression and decision tree classifiers, incorporates SMOTE for handling class imbalance, and includes risk scoring and business insights.

---

## Problem Statement

Credit card fraud causes massive financial losses. Fraudulent behavior often mimics legitimate behavior, making it difficult to detect. This project uses a dataset of ~6.3 million transactions to:

- Detect fraudulent patterns
- Fraud and Risk analysis
- Visualize risk trends
- Predict if a transaction is fraud
- Recommend strategic actions for risk mitigation

---

## Features

- Exploratory Data Analysis (EDA)
  - Fraud vs Non-Fraud pie chart
  - Correlation heatmap
  - Hourly fraud trend
- Risk Analysis
  - Fraud by transaction type
  - Boxplot of amount vs fraud
  - Risk scoring based on balance and amount behavior
- ML Prediction
  - Choose between Logistic Regression or Decision Tree
  - Predict fraud probability from transaction details
- Strategic Business Insights
  - Actionable patterns for risk reduction

---

## How to Run

### 1. Clone this Repository

```bash
git clone https://github.com/your-username/credit-card-fraud-dashboard.git
cd credit-card-fraud-dashboard
```

### 2. Install Dependencies

Make sure Python ≥3.8 is installed. Then run:

```bash
pip install -r requirements.txt
```

### 3. Add Dataset and Models

- Place `AIML_Dataset.pkl` (pickled dataset) in the root folder
- Place `fraud_model_logreg.pkl` and `fraud_model_tree.pkl` (pretrained models)

### 4. Launch the Dashboard

```bash
streamlit run dashboard.py
```

---

## Requirements

See `requirements.txt`:

```
streamlit==1.33.0
pandas==2.2.1
numpy==1.26.4
matplotlib==3.8.4
seaborn==0.13.2
scikit-learn==1.3.2
joblib==1.3.2
```

---

## Example

 
Include a screenshot of your dashboard to show how it looks in action.

---

## Future Enhancements

- Add XGBoost and Isolation Forest models
- Deploy to Streamlit Cloud
- Integrate with real-time data feeds for monitoring

---

## Author


[LinkedIn](https://www.linkedin.com/in/joan-of-arc-xavier-59a660277/) 

---

## Disclaimer

This project is for educational purposes. The data set is taken from kaggle
https://www.kaggle.com/datasets/amanalisiddiqui/fraud-detection-dataset?resource=download
