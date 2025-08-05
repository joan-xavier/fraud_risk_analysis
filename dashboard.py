import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, roc_curve, roc_auc_score

st.set_page_config(layout="wide")
st.title("\U0001F50E Credit Card Fraud & Risk Analytics Dashboard")

@st.cache_data
def load_data():
    df = pd.read_pickle("AIML_Dataset.pkl")
    df['hour'] = df['step'] % 24
    df['balance_diffOrg'] = df['oldbalanceOrg'] - df['newbalanceOrig']
    df['balance_diffDest'] = df['oldbalanceDest'] - df['newbalanceDest']
    df['riskScore'] = (
        (df['amount'] > 10000).astype(int) +
        (df['oldbalanceOrg'] == 0).astype(int) +
        (df['type'].isin(['TRANSFER', 'CASH_OUT'])).astype(int)
    )
    return df

df = load_data()

# Sidebar input
st.sidebar.title("\U0001F4E5 Input Transaction Details")
tx_type = st.sidebar.selectbox("Transaction Type", df['type'].unique())
amount = st.sidebar.slider("Transaction Amount", 0, 100000, 5000)
oldbalanceOrg = st.sidebar.slider("Old Balance Origin", 0, 100000, 5000)
newbalanceOrig = st.sidebar.slider("New Balance Origin", 0, 100000, 5000)
oldbalanceDest = st.sidebar.slider("Old Balance Destination", 0, 100000, 5000)
newbalanceDest = st.sidebar.slider("New Balance Destination", 0, 100000, 5000)

# EDA Visualizations
st.header("1\ufe0f\ufe0f Exploratory Data Analysis (EDA)")
if st.checkbox("Show EDA Charts", value=False):
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Transaction Type Distribution")
        fig1, ax1 = plt.subplots()
        df['type'].value_counts().plot(kind='bar', ax=ax1, color='skyblue')
        ax1.set_title("Distribution of Transaction Types")
        st.pyplot(fig1)
        st.markdown("**Insight:** Majority of transactions are of types PAYMENT and TRANSFER. Fraud occurs predominantly in TRANSFER and CASH_OUT.")

    with col2:
        st.subheader("Fraud Distribution")
        fig2, ax2 = plt.subplots()
        df['isFraud'].value_counts().plot(kind='pie', labels=['Non-Fraud', 'Fraud'], autopct='%1.1f%%', ax=ax2)
        ax2.set_ylabel('')
        ax2.set_title("Fraud vs Non-Fraud Ratio")
        st.pyplot(fig2)
        st.markdown("**Insight:** Data is highly imbalanced with only 0.13% fraud cases.")

    st.subheader("Feature Correlation Heatmap")
    numerical_cols = ['amount', 'oldbalanceOrg', 'newbalanceOrig', 'oldbalanceDest', 'newbalanceDest']
    corr = df[numerical_cols + ['isFraud']].corr()
    fig4, ax4 = plt.subplots()
    sns.heatmap(corr, annot=True, cmap='coolwarm', ax=ax4)
    st.pyplot(fig4)
    st.markdown("**Insight:** Strong correlation observed between balance changes and fraudulent activity.")

# Risk Indicators
st.header("2\ufe0f\ufe0f Fraud & Risk Analysis")
if st.checkbox("Show Risk Analysis", value=False):
    col3, col4 = st.columns(2)
    with col3:
        st.subheader("Fraud Rate by Transaction Type")
        fraud_rate_by_type = df.groupby('type')['isFraud'].mean().reset_index()
        fig5, ax5 = plt.subplots()
        ax5.bar(fraud_rate_by_type['type'], fraud_rate_by_type['isFraud'], color='orange')
        ax5.set_ylabel("Fraud Rate")
        st.pyplot(fig5)
        st.markdown("**Insight:** TRANSFER transactions have the highest fraud rate (0.76%) followed by CASH_OUT.")

    with col4:
        st.subheader("Box Plot: Amount by Fraud Status (< 50k)")
        fig6, ax6 = plt.subplots()
        sns.boxplot(x='isFraud', y='amount', data=df[df['amount'] < 50000], ax=ax6, showfliers=False)
        ax6.set_xticklabels(['Non-Fraud', 'Fraud'])
        st.pyplot(fig6)
        st.markdown("**Insight:** Fraudulent transactions generally have higher median amounts than non-fraud ones.")

    st.subheader("Hourly Fraud Count")
    fig7, ax7 = plt.subplots()
    sns.histplot(df[df['isFraud'] == 1]['hour'], bins=24, ax=ax7)
    ax7.set_title('Hourly Fraud Frequency')
    st.pyplot(fig7)
    st.markdown("**Insight:** Fraudulent transactions occur throughout the day but are more concentrated during working hours.")

# Prediction Section
st.header("3\ufe0f\ufe0f Predict Fraud from Input")
user_input = pd.DataFrame([{
    'type': tx_type,
    'amount': amount,
    'oldbalanceOrg': oldbalanceOrg,
    'newbalanceOrig': newbalanceOrig,
    'oldbalanceDest': oldbalanceDest,
    'newbalanceDest': newbalanceDest
}])

model_choice = st.selectbox("Choose Model", ["Logistic Regression", "Decision Tree"])
model_file = "fraud_model_logreg.pkl" if model_choice == "Logistic Regression" else "fraud_model_tree.pkl"
model = joblib.load(model_file)
prediction = model.predict(user_input)[0]
pred_proba = model.predict_proba(user_input)[0][1] if hasattr(model, "predict_proba") else None
result_label = "\u26a0\ufe0f Fraudulent Transaction" if prediction == 1 else "\u2705 Legitimate Transaction"

st.subheader("\U0001F50D Prediction Result:")
st.markdown(f"**{result_label}** with probability **{pred_proba:.2%}**" if pred_proba else f"**{result_label}**")

# Risk Scoring
st.header("4\ufe0f\ufe0f Transaction Risk Scoring")
if st.checkbox("Show Sample Risk Scoring", value=False):
    st.markdown("Accounts with higher risk scores (2 or 3) may need investigation.")
    st.dataframe(df[['type', 'amount', 'oldbalanceOrg', 'newbalanceOrig', 'riskScore']].sample(5))

# Business Insights
st.header("5\ufe0f\ufe0f Strategic Business Insights")
st.markdown("""
- Fraud is concentrated in TRANSFER and CASH_OUT transactions. Additional verification for these types is advised.
- Most frauds occur when the source account has a non-zero old balance and a zero new balance — flag such behavior.
- Transactions with high amounts are more prone to fraud. Set upper thresholds with alerts for review.
- Fraud often occurs within certain hours. Real-time monitoring and flagging can be adjusted to match those hours.
- Consider flagging transactions with high riskScore and adding manual verification.
- Use anomaly detection (like Isolation Forest) to detect hidden patterns not captured in labeled data.
- Business should monitor frequently involved fraudulent accounts and potentially block/review them.
""")

st.caption("Developed for credit card fraud risk analysis and real-time prediction using ML.")
