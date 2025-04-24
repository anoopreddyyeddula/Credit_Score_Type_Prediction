import pandas as pd
import joblib
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import streamlit as st
import plotly.express as px
import numpy as np

# Configure Streamlit with custom theme
st.set_page_config(
    page_title="Credit Score Prediction",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main {
        padding: 2rem;
    }
    .stButton>button {
        width: 100%;
        background-color: #ff4b4b;
        color: white;
        height: 3rem;
        margin-top: 2rem;
    }
    .sidebar .stButton>button {
        background-color: #4CAF50;
    }
    </style>
    """, unsafe_allow_html=True)

# Load data and model
@st.cache_resource
def load_data_and_model():
    df = pd.read_csv("train.csv")
    model = joblib.load('rf_model.pkl')
    
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

def get_credit_score_description(score):
    descriptions = {
        "Poor": """
            📉 A poor credit score indicates significant credit risks:
            - Limited access to loans and credit cards
            - Higher interest rates
            - May require secured credit cards
            - Focus on improving payment history
            """,
        "Standard": """
            📊 A standard credit score suggests moderate creditworthiness:
            - Access to standard credit cards
            - Moderate interest rates
            - Room for improvement
            - Continue maintaining timely payments
            """,
        "Average": """
            📊 An average credit score suggests moderate creditworthiness:
            - Access to standard credit cards
            - Moderate interest rates
            - Room for improvement
            - Continue maintaining timely payments
            """,
        "Good": """
            📈 A good credit score demonstrates strong creditworthiness:
            - Access to premium credit cards
            - Better interest rates
            - Higher credit limits
            - Excellent financial standing
            """
    }
    return descriptions.get(score, "")

def create_radar_chart(input_data, df):
    # Prepare data for radar chart
    numeric_cols = ["Annual_Income", "Monthly_Inhand_Salary", "Interest_Rate", 
                   "Outstanding_Debt", "Monthly_Balance"]
    
    # Normalize the values
    max_values = df[numeric_cols].max()
    normalized_values = input_data[numeric_cols].iloc[0] / max_values * 100
    
    fig = px.line_polar(
        r=normalized_values.values,
        theta=numeric_cols,
        line_close=True,
        range_r=[0, 100],
        title="Financial Profile Analysis"
    )
    fig.update_traces(fill='toself')
    return fig

df, model, pipeline = load_data_and_model()

def main():
    st.title("💳 Credit Score Prediction")
    
    # Create two columns for the main layout
    col1, col2 = st.columns([2, 3])
    
    with col1:
        st.markdown("""
        ### Enter Your Financial Details
        Please provide accurate information for better prediction.
        """)
        
        # Input form with better organization
        with st.form("prediction_form"):
            annual_income = st.number_input("Annual Income ($)", 0, 10000000, step=1000)
            monthly_salary = st.number_input("Monthly Salary ($)", 0, 1000000, step=100)
            
            st.subheader("Banking Details")
            col3, col4 = st.columns(2)
            with col3:
                num_accounts = st.number_input("Bank Accounts", 0, 20, step=1)
                num_cards = st.number_input("Credit Cards", 0, 15, step=1)
            with col4:
                num_loans = st.number_input("Number of Loans", 0, 10, step=1)
                credit_mix = st.selectbox("Credit Mix", df["Credit_Mix"].unique())
            
            st.subheader("Payment History")
            col5, col6 = st.columns(2)
            with col5:
                delay_days = st.number_input("Delay in Days", 0, 365, step=1)
                delayed_payments = st.number_input("Delayed Payments", 0, 50, step=1)
            with col6:
                interest_rate = st.number_input("Interest Rate (%)", 0, 100, step=1)
                credit_history = st.number_input("Credit History (months)", 0, 360, step=1)
            
            st.subheader("Financial Status")
            outstanding_debt = st.number_input("Outstanding Debt ($)", 0, 10000000, step=1000)
            monthly_balance = st.number_input("Monthly Balance ($)", 0, 1000000, step=100)
            
            submitted = st.form_submit_button("Predict Credit Score")
    
    with col2:
        if submitted:
            # Input data
            input_data = pd.DataFrame({
                "Annual_Income": [annual_income],
                "Monthly_Inhand_Salary": [monthly_salary],
                "Num_Bank_Accounts": [num_accounts],
                "Num_Credit_Card": [num_cards],
                "Interest_Rate": [interest_rate],
                "Num_of_Loan": [num_loans],
                "Delay_from_due_date": [delay_days],
                "Num_of_Delayed_Payment": [delayed_payments],
                "Credit_Mix": [credit_mix],
                "Outstanding_Debt": [outstanding_debt],
                "Credit_History_Age": [credit_history],
                "Monthly_Balance": [monthly_balance],
            })
            
            # Get prediction
            credit_score = pipeline.predict(input_data)[0]
            
            # Display result with enhanced UI
            st.markdown("## Analysis Results")
            
            # Credit Score Display
            score_colors = {
                "Poor": "red",
                "Standard": "orange",
                "Average": "orange",
                "Good": "green"
            }
            st.markdown(f"""
                <div style='background-color: {score_colors[credit_score]}; 
                            padding: 20px; 
                            border-radius: 10px; 
                            color: white; 
                            text-align: center;'>
                    <h2>Credit Score: {credit_score}</h2>
                </div>
                """, unsafe_allow_html=True)
            
            # Display description
            st.markdown("### What This Means")
            st.markdown(get_credit_score_description(credit_score))
            
            # Display radar chart
            st.plotly_chart(create_radar_chart(input_data, df))
            
            # Key Metrics
            st.markdown("### Key Financial Metrics")
            metrics_col1, metrics_col2, metrics_col3 = st.columns(3)
            with metrics_col1:
                st.metric("Debt to Income Ratio", 
                         f"{(outstanding_debt/(annual_income/12)*100):.1f}%")
            with metrics_col2:
                st.metric("Credit Card Utilization",
                         f"{num_cards} cards")
            with metrics_col3:
                st.metric("Payment Delay Rate",
                         f"{(delayed_payments/12*100):.1f}%")

if __name__ == "__main__":
    main()
