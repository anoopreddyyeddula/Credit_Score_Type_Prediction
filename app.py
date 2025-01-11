import pandas as pd
import joblib
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import streamlit as st

# Configure Streamlit
st.set_page_config(page_title="Credit Score Prediction", layout="centered")

# Load data and model
@st.cache_resource
def load_data_and_model():
    df = pd.read_csv("train.csv")
    model = joblib.load('rf_model.pkl')

    # Preprocessor and pipeline setup
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), ["Annual_Income", "Monthly_Inhand_Salary", 
                       "Num_Bank_Accounts", "Num_Credit_Card", 
                       "Interest_Rate", "Num_of_Loan", 
                       "Delay_from_due_date", "Num_of_Delayed_Payment", 
                        "Outstanding_Debt", 
                       "Credit_History_Age", "Monthly_Balance"]),
            ("cat", OneHotEncoder(), ["Credit_Mix"])
        ]
    )
    pipeline = Pipeline(steps=[("preprocessor", preprocessor), ("classifier", model)])
    pipeline.fit(df[["Annual_Income", "Monthly_Inhand_Salary", 
                       "Num_Bank_Accounts", "Num_Credit_Card", 
                       "Interest_Rate", "Num_of_Loan", 
                       "Delay_from_due_date", "Num_of_Delayed_Payment", 
                       "Credit_Mix", "Outstanding_Debt", 
                       "Credit_History_Age", "Monthly_Balance"]],
                  df["Credit_Score"])
    
    return df, model, pipeline

df, model, pipeline = load_data_and_model()

# Prediction function
def credit_score_classification(pipeline, input_data):
    prediction = pipeline.predict(input_data)[0]
    return prediction

# Main app logic
def main():
    st.title("\U0001F4B8 Credit Score Prediction")
    st.write("""
    Predict your credit score by entering the relevant financial details below. 
    The prediction is based on a trained model using historical data.
    """)

    # Input form with sliders and better UI
    st.sidebar.header("Input Parameters")
    annual_income = st.sidebar.slider("Annual Income", 0, 100000000, step=5000)
    monthly_inhand_salary = st.sidebar.slider("Monthly Inhand Salary", 0, 100000000, step=1000)
    num_bank_accounts = st.sidebar.slider("Number of Bank Accounts", 0, 20, step=1)
    num_credit_card = st.sidebar.slider("Number of Credit Cards", 0, 15, step=1)
    interest_rate = st.sidebar.slider("Interest Rate (%)", 0, 100, step=1)
    num_of_loan = st.sidebar.slider("Number of Loans", 0, 10, step=1)
    delay_from_due_date = st.sidebar.slider("Delay From Due Date (days)", 0, 365, step=1)
    num_delay_from_due_date = st.sidebar.slider("Number of Delayed Payments", 0, 50, step=1)
    credit_mix = st.sidebar.selectbox("Credit Mix", df["Credit_Mix"].unique())
    outstanding_debt = st.sidebar.slider("Outstanding Debt", 0, 100000000, step=1000)
    credit_history_age = st.sidebar.slider("Credit History Age (months)", 0, 360, step=1)
    monthly_balance = st.sidebar.slider("Monthly Balance", 0, 100000000, step=1000)

    # Input data
    input_data = pd.DataFrame({
        "Annual_Income": [annual_income],
        "Monthly_Inhand_Salary": [monthly_inhand_salary],
        "Num_Bank_Accounts": [num_bank_accounts],
        "Num_Credit_Card": [num_credit_card],
        "Interest_Rate": [interest_rate],
        "Num_of_Loan": [num_of_loan],
        "Delay_from_due_date": [delay_from_due_date],
        "Num_of_Delayed_Payment": [num_delay_from_due_date],
        "Credit_Mix": [credit_mix],
        "Outstanding_Debt": [outstanding_debt],
        "Credit_History_Age": [credit_history_age],
        "Monthly_Balance": [monthly_balance],
    })

    if st.sidebar.button("Predict Credit Score"):
        credit_score = credit_score_classification(pipeline, input_data)

        # Display the result
        st.markdown("### Prediction Result")
        if credit_score == "Poor":
            st.error(f"Your predicted credit score is **{credit_score}**")
        elif credit_score == "Average":
            st.warning(f"Your predicted credit score is **{credit_score}**")
        else:
            st.success(f"Your predicted credit score is **{credit_score}**")

        # Display input summary
        st.markdown("### Input Summary")
        st.write(input_data)

if __name__ == "__main__":
    main()
