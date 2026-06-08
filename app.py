import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go

# ==========================================================
# PAGE CONFIG
# ==========================================================

st.set_page_config(
    page_title="Customer Churn Prediction Dashboard",
    page_icon="📊",
    layout="wide"
)

# ==========================================================
# CUSTOM CSS
# ==========================================================

st.markdown("""
<style>

.stApp{
    background: linear-gradient(
        135deg,
        #0D0D0D,
        #1A1A1A
    );
    color:white;
}

.main-title{
    text-align:center;
    font-size:3.3rem;
    font-weight:800;
    color:#FF4D8D;
}

.subtitle{
    text-align:center;
    color:#FFB6C1;
    font-size:1.2rem;
    margin-bottom:20px;
}

.hero-card{
    background:#1A1A1A;
    border:1px solid #FF4D8D;
    border-radius:20px;
    padding:25px;
    margin-bottom:20px;
}

.metric-card{
    background:#1A1A1A;
    border:1px solid #FF4D8D;
    border-radius:15px;
    padding:15px;
}

div[data-testid="metric-container"]{
    background:#1A1A1A;
    border:1px solid #FF4D8D;
    border-radius:15px;
    padding:15px;
}

.stButton > button{
    width:100%;
    height:60px;
    border:none;
    border-radius:15px;
    background:linear-gradient(
        90deg,
        #FF4D8D,
        #FF79A8
    );
    color:white;
    font-size:20px;
    font-weight:bold;
}

.stButton > button:hover{
    transform:scale(1.02);
}

.sidebar-title{
    color:#FF4D8D;
    font-size:24px;
    font-weight:bold;
}

.block-container{
    padding-top:1rem;
}

</style>
""", unsafe_allow_html=True)

# ==========================================================
# LOAD MODEL FILES
# ==========================================================

try:

    model = joblib.load(
        "models/churn_model.pkl"
    )

    scaler = joblib.load(
        "models/scaler.pkl"
    )

    feature_names = joblib.load(
        "models/feature_names.pkl"
    )

except Exception as e:

    st.error(
        f"Error loading model files: {e}"
    )

    st.stop()

# ==========================================================
# SIDEBAR
# ==========================================================

with st.sidebar:

    st.markdown(
        '<p class="sidebar-title">📋 Dashboard Menu</p>',
        unsafe_allow_html=True
    )

    st.markdown("---")

    st.write("🤖 Model: Logistic Regression")
    st.write("📈 Accuracy: 80.67%")
    st.write("🎯 ROC-AUC: 84.29%")
    st.write("👥 Customers: 7043")
    st.write("📉 Churn Rate: 26.5%")

    st.markdown("---")

    st.info(
        """
        💡 Goal

        Predict customers
        likely to churn
        and improve
        retention strategies.
        """
    )

# ==========================================================
# HERO SECTION
# ==========================================================

st.markdown(
    """
    <p class='main-title'>
    📊 Customer Churn Prediction System
    </p>
    """,
    unsafe_allow_html=True
)

st.markdown(
    """
    <p class='subtitle'>
    AI Powered Customer Retention Analytics Dashboard
    </p>
    """,
    unsafe_allow_html=True
)

st.markdown(
    """
    <div class='hero-card'>

    <h3>🚀 Predict Customer Churn Before It Happens</h3>

    <br>

    This dashboard uses Machine Learning to identify
    customers likely to leave.

    <br><br>

    ✔ Logistic Regression Model

    <br>

    ✔ Telco Customer Churn Dataset

    <br>

    ✔ Retention Focused Insights

    <br>

    ✔ Real Time Churn Prediction

    </div>
    """,
    unsafe_allow_html=True
)

# ==========================================================
# KPI CARDS
# ==========================================================

k1, k2, k3, k4 = st.columns(4)

with k1:
    st.metric(
        "Accuracy",
        "80.67%"
    )

with k2:
    st.metric(
        "ROC-AUC",
        "84.29%"
    )

with k3:
    st.metric(
        "Customers",
        "7043"
    )

with k4:
    st.metric(
        "Churn Rate",
        "26.5%"
    )

st.markdown("---")

# ==========================================================
# CUSTOMER INFORMATION
# ==========================================================

st.subheader("👤 Customer Information")

c1, c2 = st.columns(2)

with c1:

    tenure = st.slider(
        "Tenure Months",
        min_value=0,
        max_value=72,
        value=12
    )

    gender = st.selectbox(
        "Gender",
        ["Female", "Male"]
    )

    senior_citizen = st.selectbox(
        "Senior Citizen",
        ["No", "Yes"]
    )

with c2:

    partner = st.selectbox(
        "Partner",
        ["No", "Yes"]
    )

    dependents = st.selectbox(
        "Dependents",
        ["No", "Yes"]
    )

    contract = st.selectbox(
        "Contract Type",
        [
            "Month-to-month",
            "One year",
            "Two year"
        ]
    )

st.markdown("---")

# ==========================================================
# BILLING INFORMATION
# ==========================================================

st.subheader("💳 Billing Information")

b1, b2 = st.columns(2)

with b1:

    monthly_charges = st.number_input(
        "Monthly Charges",
        min_value=0.0,
        max_value=200.0,
        value=70.0
    )

    total_charges = st.number_input(
        "Total Charges",
        min_value=0.0,
        value=1000.0
    )

with b2:

    cltv = st.number_input(
        "Customer Lifetime Value (CLTV)",
        min_value=0,
        value=4000
    )

    payment_method = st.selectbox(
        "Payment Method",
        [
            "Bank transfer (automatic)",
            "Credit card (automatic)",
            "Electronic check",
            "Mailed check"
        ]
    )

st.markdown("---")

# ==========================================================
# SERVICE INFORMATION
# ==========================================================

st.subheader("📡 Service Information")

s1, s2 = st.columns(2)

with s1:

    internet_service = st.selectbox(
        "Internet Service",
        [
            "DSL",
            "Fiber optic",
            "No"
        ]
    )

    phone_service = st.selectbox(
        "Phone Service",
        [
            "No",
            "Yes"
        ]
    )

    multiple_lines = st.selectbox(
        "Multiple Lines",
        [
            "No",
            "Yes",
            "No phone service"
        ]
    )

    online_security = st.selectbox(
        "Online Security",
        [
            "No",
            "Yes",
            "No internet service"
        ]
    )

    online_backup = st.selectbox(
        "Online Backup",
        [
            "No",
            "Yes",
            "No internet service"
        ]
    )

with s2:

    device_protection = st.selectbox(
        "Device Protection",
        [
            "No",
            "Yes",
            "No internet service"
        ]
    )

    tech_support = st.selectbox(
        "Tech Support",
        [
            "No",
            "Yes",
            "No internet service"
        ]
    )

    streaming_tv = st.selectbox(
        "Streaming TV",
        [
            "No",
            "Yes",
            "No internet service"
        ]
    )

    streaming_movies = st.selectbox(
        "Streaming Movies",
        [
            "No",
            "Yes",
            "No internet service"
        ]
    )

    paperless_billing = st.selectbox(
        "Paperless Billing",
        [
            "No",
            "Yes"
        ]
    )

st.markdown("---")

# ==========================================================
# PREDICT BUTTON
# ==========================================================

predict_btn = st.button(
    "🚀 Predict Churn",
    use_container_width=True
)

# ==========================================================
# FEATURE VECTOR
# ==========================================================

sample_customer = [0] * 31

# Numerical Features

sample_customer[0] = tenure
sample_customer[1] = monthly_charges
sample_customer[2] = total_charges
sample_customer[3] = cltv

# Gender

if gender == "Male":
    sample_customer[4] = 1

# Senior Citizen

if senior_citizen == "Yes":
    sample_customer[5] = 1

# Partner

if partner == "Yes":
    sample_customer[6] = 1

# Dependents

if dependents == "Yes":
    sample_customer[7] = 1

# Phone Service

if phone_service == "Yes":
    sample_customer[8] = 1

# Multiple Lines

if multiple_lines == "No phone service":
    sample_customer[9] = 1

elif multiple_lines == "Yes":
    sample_customer[10] = 1

# Internet Service

if internet_service == "Fiber optic":
    sample_customer[11] = 1

elif internet_service == "No":
    sample_customer[12] = 1

# Online Security

if online_security == "No internet service":
    sample_customer[13] = 1

elif online_security == "Yes":
    sample_customer[14] = 1

# Online Backup

if online_backup == "No internet service":
    sample_customer[15] = 1

elif online_backup == "Yes":
    sample_customer[16] = 1

# Device Protection

if device_protection == "No internet service":
    sample_customer[17] = 1

elif device_protection == "Yes":
    sample_customer[18] = 1

# Tech Support

if tech_support == "No internet service":
    sample_customer[19] = 1

elif tech_support == "Yes":
    sample_customer[20] = 1

# Streaming TV

if streaming_tv == "No internet service":
    sample_customer[21] = 1

elif streaming_tv == "Yes":
    sample_customer[22] = 1

# Streaming Movies

if streaming_movies == "No internet service":
    sample_customer[23] = 1

elif streaming_movies == "Yes":
    sample_customer[24] = 1

# Contract

if contract == "One year":
    sample_customer[25] = 1

elif contract == "Two year":
    sample_customer[26] = 1

# Paperless Billing

if paperless_billing == "Yes":
    sample_customer[27] = 1

# Payment Method

if payment_method == "Credit card (automatic)":
    sample_customer[28] = 1

elif payment_method == "Electronic check":
    sample_customer[29] = 1

elif payment_method == "Mailed check":
    sample_customer[30] = 1

# ==========================================================
# PREDICTION ENGINE
# ==========================================================

if predict_btn:

    customer_df = pd.DataFrame(
        [sample_customer],
        columns=feature_names
    )

    customer_scaled = scaler.transform(
        customer_df
    )

    prediction = model.predict(
        customer_scaled
    )

    probability = model.predict_proba(
        customer_scaled
    )[0][1]

    # ======================================================
    # RISK LEVEL
    # ======================================================

    if probability < 0.30:
        risk = "🟢 Low Risk"

    elif probability < 0.70:
        risk = "🟡 Medium Risk"

    else:
        risk = "🔴 High Risk"

    # ======================================================
    # SAVE PREDICTION HISTORY
    # ======================================================

    history_file = "predictions.csv"

    record = {
        "Timestamp":
        datetime.now().strftime(
            "%Y-%m-%d %H:%M:%S"
        ),

        "Tenure":
        tenure,

        "Contract":
        contract,

        "Prediction":
        "Churn"
        if prediction[0] == 1
        else "Stay",

        "Probability (%)":
        round(
            probability * 100,
            2
        )
    }

    record_df = pd.DataFrame(
        [record]
    )

    if os.path.exists(
        history_file
    ):

        record_df.to_csv(
            history_file,
            mode="a",
            header=False,
            index=False
        )

    else:

        record_df.to_csv(
            history_file,
            index=False
        )

    st.markdown("---")

    # ======================================================
    # RISK DISTRIBUTION
    # ======================================================

    st.subheader(
        "📊 Churn Risk Distribution"
    )

    churn_percent = (
        probability * 100
    )

    stay_percent = (
        100 - churn_percent
    )

    c1, c2 = st.columns(2)

    with c1:
        st.metric(
            "Stay Probability",
            f"{stay_percent:.2f}%"
        )

    with c2:
        st.metric(
            "Churn Probability",
            f"{churn_percent:.2f}%"
        )

    st.progress(
        float(probability)
    )

    # ======================================================
    # PREDICTION RESULT
    # ======================================================

    st.markdown("---")

    st.subheader(
        "📈 Prediction Result"
    )

    if prediction[0] == 0:

        st.success(
            "🟢 Customer Likely To Stay"
        )

        recommendation = """
        • Continue engagement

        • Offer loyalty rewards

        • Monitor customer activity

        • Maintain service quality
        """

    else:

        st.error(
            "🔴 Customer Likely To Churn"
        )

        recommendation = """
        • Contact customer immediately

        • Offer retention discount

        • Provide dedicated support

        • Upgrade service package
        """

    st.info(
        f"Risk Level: {risk}"
    )

    st.markdown(
        f"""
        ### 💡 Recommendation

        {recommendation}
        """
    )

# ==========================================================
# FEATURE IMPORTANCE
# ==========================================================

st.markdown("---")

st.subheader(
    "📈 Feature Importance Analysis"
)

importance_df = pd.DataFrame({

    "Feature":[
        "Dependents_Yes",
        "Contract_Two year",
        "Contract_One year",
        "Online Security_Yes",
        "Tech Support_Yes",
        "Internet Service_Fiber optic",
        "Electronic Check"
    ],

    "Importance":[
        1.69,
        1.17,
        0.74,
        0.43,
        0.42,
        0.57,
        0.36
    ]
})

fig = px.bar(
    importance_df,
    x="Importance",
    y="Feature",
    orientation="h",
    title="Top Features Affecting Churn"
)

fig.update_layout(
    paper_bgcolor="#1A1A1A",
    plot_bgcolor="#1A1A1A",
    font_color="white"
)

st.plotly_chart(
    fig,
    use_container_width=True
)

# ==========================================================
# INSIGHTS
# ==========================================================

st.info("""
📌 Key Insights

• Customers with Dependents are less likely to churn.

• Long-term contracts significantly reduce churn.

• Online Security and Tech Support improve retention.

• Fiber Optic users show higher churn tendency.

• Electronic Check customers have elevated churn risk.
""")

# ==========================================================
# PREDICTION HISTORY
# ==========================================================

st.markdown("---")

st.subheader(
    "📜 Prediction History"
)

history_file = "predictions.csv"

if os.path.exists(
    history_file
):

    history_df = pd.read_csv(
        history_file
    )

    st.dataframe(
        history_df.tail(20),
        use_container_width=True
    )

    st.download_button(
        "📥 Download Prediction History",
        history_df.to_csv(
            index=False
        ),
        file_name=
        "prediction_history.csv",
        mime="text/csv"
    )

else:

    st.info(
        "No prediction history available."
    )

# ==========================================================
# PDF REPORT PLACEHOLDER
# ==========================================================

st.markdown("---")

st.subheader(
    "📄 Report Export"
)

st.info(
    """
    PDF Export Module
    will be added in Version 2.0
    """
)

# ==========================================================
# FOOTER
# ==========================================================

st.markdown("---")

st.markdown(
    """
    <div style='
    text-align:center;
    color:#FFB6C1;
    font-size:14px;'>

    Customer Churn Prediction System

    <br>

    AI Powered Customer Retention Analytics Dashboard

    <br><br>

    Built using:
    Streamlit • Scikit-Learn • Plotly

    </div>
    """,
    unsafe_allow_html=True
)