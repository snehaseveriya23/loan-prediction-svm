import streamlit as st
import numpy as np
import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt


# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="Loan Prediction System",
    page_icon="🏦",
    layout="centered",
    initial_sidebar_state="collapsed"
)


# =========================================================
# CUSTOM CSS - BLUE BANKING THEME
# =========================================================

st.markdown("""
<style>

    /* ---------- MAIN BACKGROUND ---------- */

    .stApp {
        background-color: #f4f7fb;
    }

    .block-container {
        max-width: 950px;
        padding-top: 2rem;
        padding-bottom: 3rem;
    }


    /* ---------- HEADER ---------- */

    .bank-header {
        background: linear-gradient(135deg, #071d35, #0d3b66, #155d9a);
        padding: 30px 34px;
        border-radius: 16px;
        color: white;
        margin-bottom: 25px;
        box-shadow: 0 8px 24px rgba(13, 59, 102, 0.18);
    }

    .bank-name {
        font-size: 14px;
        font-weight: 600;
        letter-spacing: 1.8px;
        color: #b9d9f5;
        margin-bottom: 7px;
    }

    .bank-title {
        font-size: 30px;
        font-weight: 700;
        color: white;
        margin-bottom: 8px;
    }

    .bank-subtitle {
        font-size: 15px;
        color: #d9eafa;
    }


    /* ---------- SECTION CARDS ---------- */

    .section-card {
        background-color: white;
        padding: 22px 25px 12px 25px;
        border-radius: 14px;
        margin-bottom: 20px;
        border: 1px solid #dfe7f0;
        box-shadow: 0 4px 14px rgba(20, 50, 80, 0.06);
    }

    .section-title {
        font-size: 19px;
        font-weight: 650;
        color: #0b3157;
        margin-bottom: 15px;
    }


    /* ---------- BUTTON ---------- */

    div.stButton > button {
        width: 100%;
        background: linear-gradient(135deg, #0d3b66, #1769aa);
        color: white;
        border: none;
        border-radius: 9px;
        padding: 12px 20px;
        font-size: 16px;
        font-weight: 600;
        box-shadow: 0 4px 12px rgba(13, 59, 102, 0.18);
    }

    div.stButton > button:hover {
        background: linear-gradient(135deg, #082b4c, #0d5c96);
        color: white;
    }


    /* ---------- FOOTER ---------- */

    .footer {
        text-align: center;
        color: #7b8794;
        font-size: 12px;
        margin-top: 30px;
        padding-top: 15px;
    }


    /* ---------- GLOBAL TEXT VISIBILITY ---------- */

    .stApp {
        background-color: #f4f7fb;
        color: #102a43 !important;
    }

    /* Force readable text on Streamlit Cloud */
    [data-testid="stMarkdownContainer"],
    [data-testid="stMarkdownContainer"] p,
    [data-testid="stMarkdownContainer"] li,
    [data-testid="stMarkdownContainer"] h1,
    [data-testid="stMarkdownContainer"] h2,
    [data-testid="stMarkdownContainer"] h3,
    [data-testid="stMarkdownContainer"] h4 {
        color: #102a43 !important;
    }

    /* Expander content and header */
    [data-testid="stExpander"] p,
    [data-testid="stExpander"] li,
    [data-testid="stExpander"] span,
    [data-testid="stExpander"] div,
    [data-testid="stExpander"] summary {
        color: #102a43 !important;
    }

    /* Streamlit subheaders */
    [data-testid="stSubheader"] {
        color: #102a43 !important;
    }

    /* Captions */
    .stCaption,
    [data-testid="stCaptionContainer"] {
        color: #52606d !important;
    }

    /* Keep banking header text white */
    .bank-header,
    .bank-header * {
        color: white !important;
    }

    .bank-header .bank-name {
        color: #b9d9f5 !important;
    }

    .bank-header .bank-subtitle {
        color: #d9eafa !important;
    }

    /* Keep prediction result colors */
    .approved,
    .approved * {
        color: #176b36 !important;
    }

    .not-approved,
    .not-approved * {
        color: #a12626 !important;
    }

    /* ---------- INFO CARD ---------- */

    .info-card {
        background-color: #eef5fc;
        border-left: 4px solid #1769aa;
        padding: 12px 15px;
        border-radius: 7px;
        margin-top: 10px;
        color: #102a43 !important;
        font-size: 14px;
    }

    .info-card * {
        color: #102a43 !important;
    }

</style>
""", unsafe_allow_html=True)


# =========================================================
# LOAD MODEL AND SHAP BACKGROUND
# =========================================================

model = joblib.load("loan_svm_model.pkl")

shap_background = joblib.load("loan_shap_background.pkl")


# =========================================================
# FEATURE NAMES
# =========================================================

feature_names = [
    "Gender",
    "Married",
    "Dependents",
    "Education",
    "Self_Employed",
    "ApplicantIncome",
    "CoapplicantIncome",
    "LoanAmount",
    "Loan_Amount_Term",
    "Credit_History",
    "Property_Area"
]


# =========================================================
# CREATE SHAP EXPLAINER
# =========================================================

explainer = shap.LinearExplainer(
    model,
    shap_background
)


# =========================================================
# HEADER
# =========================================================

st.markdown(
    '<div class="bank-header">'
    '<div class="bank-name">LOANWISE • DIGITAL BANKING</div>'
    '<div class="bank-title">🏦 Loan Prediction System</div>'
    '<div class="bank-subtitle">'
    'Predict loan approval outcomes using machine learning and model explainability.'
    '</div>'
    '</div>',
    unsafe_allow_html=True
)


# =========================================================
# APPLICANT INFORMATION
# =========================================================

st.markdown(
    '<div class="section-card">'
    '<div class="section-title">👤 Applicant Information</div>',
    unsafe_allow_html=True
)

col1, col2 = st.columns(2)

with col1:
    gender = st.selectbox(
        "Gender",
        ["Male", "Female"]
    )

with col2:
    married = st.selectbox(
        "Marital Status",
        ["Yes", "No"]
    )


col1, col2 = st.columns(2)

with col1:
    dependents = st.selectbox(
        "Number of Dependents",
        [0, 1, 2, "3+"]
    )

with col2:
    education = st.selectbox(
        "Education",
        ["Graduate", "Not Graduate"]
    )


col1, col2 = st.columns(2)

with col1:
    self_employed = st.selectbox(
        "Self Employed",
        ["Yes", "No"]
    )

with col2:
    property_area = st.selectbox(
        "Property Area",
        ["Urban", "Semiurban", "Rural"]
    )

st.markdown("</div>", unsafe_allow_html=True)


# =========================================================
# FINANCIAL INFORMATION
# =========================================================

st.markdown(
    '<div class="section-card">'
    '<div class="section-title">💰 Financial Information</div>',
    unsafe_allow_html=True
)

col1, col2 = st.columns(2)

with col1:
    applicant_income = st.number_input(
        "Applicant Income (per month)",
        min_value=0,
        value=5000,
        step=500,
        help="Enter the applicant's monthly income."
    )

with col2:
    coapplicant_income = st.number_input(
        "Co-applicant Income (per month)",
        min_value=0,
        value=0,
        step=500,
        help="Enter the co-applicant's monthly income."
    )


col1, col2 = st.columns(2)

with col1:
    loan_amount = st.number_input(
        "Loan Amount (in thousands)",
        min_value=0,
        value=180,
        step=10,
        help="Loan amount as represented in the dataset."
    )

with col2:
    loan_term = st.number_input(
        "Loan Term (months)",
        min_value=0,
        value=360,
        step=12,
        help="Loan repayment period in months."
    )


col1, col2 = st.columns(2)

with col1:
    credit_history = st.selectbox(
        "Credit History",
        ["Yes", "No"],
        help="Whether the applicant has a satisfactory credit history."
    )

st.markdown("</div>", unsafe_allow_html=True)


# =========================================================
# MODEL INFORMATION
# =========================================================

with st.expander("ℹ️ About this prediction"):

    st.write(
        "This application uses a tuned Support Vector Machine (SVM) "
        "classification model trained on the loan prediction dataset."
    )

    st.write(
        "**Model Test Accuracy:** 81.25%"
    )

    st.caption(
        "This prediction is generated by a machine learning model "
        "and should not be treated as a guaranteed banking decision."
    )


# =========================================================
# PREDICTION BUTTON
# =========================================================

st.write("")

predict_button = st.button(
    "🔍 Predict Loan Status",
    use_container_width=True
)


# =========================================================
# PREDICTION
# =========================================================

if predict_button:

    # -----------------------------------------------------
    # ENCODING
    # -----------------------------------------------------

    gender_encoded = 1 if gender == "Male" else 0

    married_encoded = 1 if married == "Yes" else 0

    if dependents == "3+":
        dependents_encoded = 4
    else:
        dependents_encoded = dependents

    education_encoded = 1 if education == "Graduate" else 0

    self_employed_encoded = 1 if self_employed == "Yes" else 0

    credit_history_encoded = 1 if credit_history == "Yes" else 0

    property_area_encoded = {
        "Rural": 0,
        "Semiurban": 1,
        "Urban": 2
    }[property_area]


    # -----------------------------------------------------
    # INPUT DATA
    # EXACT SAME ORDER AS TRAINING DATA
    # -----------------------------------------------------

    input_data = [
        gender_encoded,
        married_encoded,
        dependents_encoded,
        education_encoded,
        self_employed_encoded,
        applicant_income,
        coapplicant_income,
        loan_amount,
        loan_term,
        credit_history_encoded,
        property_area_encoded
    ]


    # -----------------------------------------------------
    # CREATE DATAFRAME
    # -----------------------------------------------------

    input_df = pd.DataFrame(
        [input_data],
        columns=feature_names
    )


    # -----------------------------------------------------
    # MODEL PREDICTION
    # -----------------------------------------------------

    prediction = model.predict(input_df)


    # -----------------------------------------------------
    # PREDICTION RESULT
    # -----------------------------------------------------

    st.write("")

    st.subheader("Prediction Result")

    if prediction[0] == 1:

        st.success(
            "✓ Loan Likely to be Approved"
        )

        st.write(
            "Based on the information provided, the tuned SVM "
            "model predicts an approved loan outcome."
        )

    else:

        st.error(
            "✕ Loan Likely to be Not Approved"
        )

        st.write(
            "Based on the information provided, the tuned SVM "
            "model predicts a not-approved loan outcome."
        )


    # -----------------------------------------------------
    # MODEL INFORMATION
    # -----------------------------------------------------

    st.markdown(
        '<div class="info-card">'
        '<b>Model:</b> Tuned Support Vector Machine (SVM)<br>'
        '<b>Test Accuracy:</b> 81.25%'
        '</div>',
        unsafe_allow_html=True
    )


    # =====================================================
    # SHAP MODEL EXPLAINABILITY
    # =====================================================

    st.write("")

    st.subheader("🔎 Model Explainability")

    st.write(
        "SHAP explains how individual applicant features influenced "
        "the model's prediction for this application."
    )


    # Calculate SHAP values
    shap_values = explainer(input_df)


    # -----------------------------------------------------
    # SHAP WATERFALL PLOT
    # -----------------------------------------------------

    fig, ax = plt.subplots(figsize=(9, 6))

    shap.plots.waterfall(
        shap_values[0],
        max_display=11,
        show=False
    )

    st.pyplot(fig, clear_figure=True)

    plt.close(fig)


    st.caption(
        "Features pushing the prediction higher contribute toward "
        "approval, while features pushing it lower contribute toward "
        "non-approval. Larger SHAP values indicate stronger influence."
    )


# =========================================================
# FOOTER
# =========================================================

st.markdown(
    '<div class="footer">'
    'Loan Prediction System • Powered by Machine Learning & SHAP<br>'
    'For educational and demonstration purposes only.'
    '</div>',
    unsafe_allow_html=True
)
