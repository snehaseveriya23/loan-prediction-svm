# 🏦 Loan Prediction System

### SVM + Hyperparameter Tuning + SHAP Explainability + Streamlit

An end-to-end machine learning project that predicts loan approval outcomes using classification models and explains individual predictions using SHAP.

The final tuned Support Vector Machine (SVM) model is integrated into an interactive Streamlit application for real-time loan prediction and model explainability.

---

## 📌 Project Overview

Loan approval decisions depend on multiple applicant and financial attributes such as income, credit history, loan amount, education, marital status, and property area.

The objective of this project is to build a machine learning classification system that:

- Predicts whether a loan application is likely to be approved or not
- Compares multiple classification algorithms
- Uses cross-validation and hyperparameter tuning to improve model selection
- Evaluates the final model using multiple performance metrics
- Uses SHAP to explain the model's predictions
- Deploys the final model through an interactive Streamlit application

---

## 🎯 Problem Statement

Build a binary classification model that predicts the loan approval status of an applicant based on demographic, financial, and credit-related features.

### Target Variable

`Loan_Status`

- `0` → Not Approved
- `1` → Approved

---

## 🔄 Project Workflow

```text
Dataset
   ↓
Data Cleaning
   ↓
Exploratory Data Analysis
   ↓
Categorical Encoding
   ↓
Train-Test Split
   ↓
Model Training
   ├── SVM
   ├── Random Forest
   └── XGBoost
   ↓
Model Comparison
   ↓
Cross-Validation
   ↓
SVM Hyperparameter Tuning
   ↓
Final SVM Evaluation
   ↓
SHAP Explainability
   ↓
Streamlit Deployment
   ↓
Loan Prediction + Model Explanation

📊 Dataset

The project uses a loan prediction dataset containing applicant demographic, financial, and credit-related information.

Dataset Information
Original records: 614
Records after removing missing values: 480
Training records: 384
Testing records: 96
Features
Feature	Description
Gender	Applicant gender
Married	Marital status
Dependents	Number of dependents
Education	Education level
Self_Employed	Self-employment status
ApplicantIncome	Applicant income
CoapplicantIncome	Co-applicant income
LoanAmount	Requested loan amount
Loan_Amount_Term	Loan repayment term
Credit_History	Credit history indicator
Property_Area	Rural, Semiurban, or Urban
🧹 Data Preprocessing

The following preprocessing steps were performed.

1. Missing Value Handling

Rows containing missing values were removed from the dataset.

This reduced the dataset from 614 records to 480 records.

2. Target Encoding

The target variable was converted from:

N → 0
Y → 1
3. Categorical Encoding

Categorical variables were converted into numerical representations.

Gender:
Male → 1
Female → 0


Married:
Yes → 1
No → 0


Education:
Graduate → 1
Not Graduate → 0


Self_Employed:
Yes → 1
No → 0


Property_Area:
Rural → 0
Semiurban → 1
Urban → 2

For Dependents:

3+ → 4
4. Feature and Target Separation

Loan_ID and Loan_Status were removed from the feature matrix.

The remaining features were used as model inputs.

🤖 Models Compared

Three classification algorithms were initially evaluated:

Support Vector Machine (SVM)
Random Forest
XGBoost
Model Comparison
Model	Training Accuracy	Testing Accuracy
SVM	77.86%	81.25%
Random Forest	100.00%	79.17%
XGBoost	100.00%	80.21%

SVM achieved the highest testing accuracy among the three models and was selected for further cross-validation and hyperparameter tuning.

🔁 Cross-Validation

Cross-validation was performed on the SVM model using 3 folds.

The initial mean cross-validation accuracy was:

72.66%

This step was used to evaluate the model across multiple training-validation splits rather than relying only on a single train-test split.

⚙️ Hyperparameter Tuning

GridSearchCV was used to identify the best SVM hyperparameters.

Parameter Search
C:
0.01, 0.1, 1, 10, 100


Gamma:
scale, auto


Kernel:
linear, rbf
Best Parameters
C = 0.1
Gamma = scale
Kernel = linear
Best Cross-Validation Accuracy

78.13%

The tuned SVM was then evaluated on the unseen test dataset.

📈 Final Model Performance

The tuned SVM achieved:

Metric	Score
Accuracy	81.25%
Precision	80.77%
Recall	95.45%
F1 Score	87.50%

The model was additionally evaluated using a confusion matrix.

<!-- ADD SCREENSHOT HERE --> <!-- Suggested screenshot: Confusion Matrix -->
🔎 SHAP Model Explainability

To make the machine learning model more interpretable, SHAP (SHapley Additive exPlanations) was used.

SHAP helps identify how individual features influence model predictions.

Global Feature Importance

The SHAP analysis identified the following as the most influential features:

Credit_History
ApplicantIncome
Property_Area

The SHAP summary plot provides an overall view of how feature values influence the model output.

<!-- ADD SCREENSHOT HERE --> <!-- Suggested screenshot: SHAP Summary Plot from the Jupyter Notebook -->
Individual Prediction Explainability

The Streamlit application also generates a SHAP waterfall plot for an individual applicant.

This answers:

Why did the model make this particular prediction?

Features with positive SHAP contributions push the model output higher, while negative contributions push it lower.

<!-- ADD SCREENSHOT HERE --> <!-- Suggested screenshot: SHAP Model Explainability section from Streamlit -->
🖥️ Streamlit Application

The final tuned SVM model was saved using Joblib and integrated into a Streamlit application.

The application allows users to enter:

Applicant information
Financial information
Credit history
Property area
Loan details

The system then provides:

1. Loan Prediction
✓ Loan Likely to be Approved

or

✕ Loan Likely to be Not Approved
<!-- ADD SCREENSHOT HERE --> <!-- Suggested screenshot: Main Streamlit interface + prediction result -->
2. Model Information

The application displays the final model and its test accuracy.

3. SHAP Model Explainability

The application generates an individual SHAP explanation showing how the applicant's features contributed to the prediction.

🧠 Deployment Architecture
User
 ↓
Streamlit Interface
 ↓
Applicant Inputs
 ↓
Categorical Encoding
 ↓
Feature Arrangement
 ↓
Saved Tuned SVM Model
 ↓
Prediction
 ↓
SHAP Explanation
 ↓
Result + Explanation

The trained model is stored as:

loan_svm_model.pkl

The SHAP background dataset is stored as:

loan_shap_background.pkl
📁 Project Structure
loan-prediction-svm/
│
├── app.py
│
├── Loan_Prediction_System.ipynb
│
├── Loan Prediction Dataset.csv
│
├── loan_svm_model.pkl
│
├── loan_shap_background.pkl
│
├── .gitignore
│
└── README.md
🛠️ Technologies Used
Programming Language
Python
Data Processing
Pandas
NumPy
Data Visualization
Matplotlib
Seaborn
Machine Learning
Scikit-learn
SVM
Random Forest
XGBoost
Model Explainability
SHAP
Deployment
Streamlit
Model Serialization
Joblib
▶️ How to Run the Project
1. Clone the repository
git clone https://github.com/snehaseveriya23/loan-prediction-svm.git
2. Navigate to the project directory
cd loan-prediction-svm
3. Install required libraries
pip install pandas numpy matplotlib seaborn scikit-learn xgboost shap streamlit joblib
4. Run the Streamlit application
streamlit run app.py

The application will open in your browser.

📌 Key Highlights
Compared SVM, Random Forest, and XGBoost
Selected SVM based on test performance
Performed cross-validation
Performed GridSearchCV hyperparameter tuning
Achieved 81.25% test accuracy
Achieved 95.45% recall
Used SHAP for model explainability
Built an interactive Streamlit application
Integrated individual prediction explanations using SHAP
🚀 Future Improvements

Potential future improvements include:

What-if analysis for applicant inputs
Probability/risk estimation
Fairness and bias analysis
REST API integration
Model monitoring
Improved preprocessing pipeline
Cloud deployment
Automated model retraining
⚠️ Disclaimer

This project is developed for educational and demonstration purposes.

The predictions generated by the model should not be considered guaranteed banking or financial decisions.

👩‍💻 Author

Sneha Severiya

B.Tech Data Science Student

GitHub:
https://github.com/snehaseveriya23
