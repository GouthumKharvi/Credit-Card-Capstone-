# FindDefault: Prediction of Credit Card Fraud

## Introduction
Credit cards are widely used for online purchases and payments, making them a convenient tool for managing personal finances. However, this convenience comes with a risk: credit card fraud. Fraudulent activities can cause significant financial loss to both customers and financial institutions. This project, titled **FindDefault**, focuses on predicting fraudulent credit card transactions. The primary goal is to build a robust classification model that accurately distinguishes between legitimate and fraudulent transactions, thus helping credit card companies minimize losses and protect their customers.

## Dataset Description
The dataset used in this project contains credit card transactions made by European cardholders in September 2013. The dataset includes transactions from a two-day period, totaling 284,807 transactions, of which 492 are fraudulent. This results in a highly imbalanced dataset, with fraudulent transactions making up only 0.172% of all transactions. 




**Key Dataset Characteristics:**
- **Total Transactions:** 284,807
- **Fraudulent Transactions:** 492
- **Class Imbalance:** Fraudulent transactions account for 0.172% of the data.

The dataset contains various features, such as transaction amount, timestamp, and anonymized numerical features, which are derived from the original data to protect sensitive information.

## Project Pipeline
The project is structured into the following key phases:

### 1. Exploratory Data Analysis (EDA)
   - **Objective:** Gain insights into the data, identify patterns, relationships, and trends.
   - **Steps:**
     - Load the dataset and display the first few rows to understand its structure.
     - Generate summary statistics to observe central tendencies, variability, and distribution of data.
     - Visualize data distributions, correlations, and relationships using histograms, box plots, scatter plots, and heatmaps.
     - Identify any significant patterns or anomalies, such as correlations between features and the target variable (fraud).

### 2. Data Cleaning and Preprocessing
   - **Objective:** Prepare the dataset for modeling by addressing issues such as missing values, outliers, and data inconsistencies.
   - **Steps:**
     - Check for missing values and handle them if necessary (e.g., imputation or removal).
     - Identify and address outliers that could distort model performance.
     - Standardize or normalize numerical features to ensure all features contribute equally to the model.
     - Encode categorical variables, if any, using techniques like one-hot encoding or label encoding.

### 3. Handling Imbalanced Data
   - **Objective:** Address the class imbalance to prevent the model from being biased towards the majority class.
   - **Approaches:**
     - **Undersampling:** Randomly reduce the number of non-fraudulent transactions to match the count of fraudulent transactions.
     - **Oversampling:** Increase the number of fraudulent transactions to match the count of non-fraudulent transactions using techniques like Random Oversampling.
     - **SMOTE (Synthetic Minority Over-sampling Technique):** Generate synthetic samples of the minority class to balance the dataset.
     - **ADASYN (Adaptive Synthetic Sampling):** Similar to SMOTE but focuses on creating synthetic samples in regions with low density of minority class samples.

### 4. Feature Engineering
   - **Objective:** Enhance model performance by creating new features or transforming existing ones.
   - **Steps:**
     - Create new features that capture important patterns or relationships not directly available in the original dataset.
     - Apply dimensionality reduction techniques like PCA if necessary to reduce feature space and improve model efficiency.
     - Transform features to reduce skewness, normalize distributions, or handle categorical data.

### 5. Model Selection
   - **Objective:** Select appropriate machine learning algorithms for the classification task.
   - **Considered Models:**
     - **Logistic Regression:** A simple yet effective model for binary classification tasks.
     - **Decision Trees:** A model that can capture complex relationships between features.
     - **XGBoost:** A powerful ensemble method that combines multiple weak learners to form a strong predictive model.

### 6. Model Training and Validation
   - **Objective:** Train the model on the training dataset and validate its performance on the validation set.
   - **Steps:**
     - Split the dataset into training and testing sets to evaluate the model’s performance on unseen data.
     - Train the models using the training set and perform cross-validation to assess the robustness of the model.
     - Tune hyperparameters using techniques like GridSearchCV to find the optimal model parameters.

### 7. Model Evaluation
   - **Objective:** Evaluate the model's performance using various metrics and choose the best-performing model.
   - **Performance Metrics:**
     - **Accuracy:** The proportion of correctly classified transactions.
     - **Precision:** The proportion of true positive predictions among all positive predictions.
     - **Recall:** The proportion of actual fraudulent transactions that were correctly identified.
     - **F1-Score:** The harmonic mean of precision and recall, providing a balance between the two.
     - **ROC-AUC:** The area under the receiver operating characteristic curve, measuring the model’s ability to distinguish between classes.
   - **Visualization:** Use a confusion matrix to visualize the model's performance and understand the types of errors it makes.

### 8. Model Deployment
   - **Objective:** Deploy the best-performing model in a production environment for real-time fraud detection.
   - **Steps:**
     - Serialize the model using the pickle module to save it for future use.
     - Example code snippet to save the model:
       ```python
       with open('load_best_model.pkl', 'wb') as file:
           pickle.dump(load_best_model, file)
       ```
     - Deploy the model to a production environment where it can be used to make real-time predictions on new transactions.

## Results and Insights
   - **Summary of Findings:** Present key insights from the EDA phase, such as patterns or correlations that significantly impact the prediction of fraud.
   - **Feature Importance:** Identify and discuss the most important features that influence the model’s decisions.
   - **Model Performance:** Report the final model’s performance on the test set, including comparisons with baseline models. Highlight the model’s strengths and any potential limitations.
   - **Final Model:** The Logistic Regression model with SMOTE balancing was selected for its excellent performance, achieving an ROC-AUC score of 0.99 on the train set and 0.97 on the test set.


**Streamlit Integration**
Streamlit is used to create an interactive web application for real-time fraud detection. The Streamlit app provides a user-friendly interface to input transaction data and receive predictions.

**Usage of Streamlit**
**Setup Streamlit App**

Create a **streamlit_app.py** file for the application interface.
Implement functionality to input transaction data and display predictions.

**Streamlit Application Code**
import streamlit as st
import pickle
import pandas as pd
from sklearn.preprocessing import StandardScaler

# Load pre-trained model
with open('load_best_model.pkl', 'rb') as file:
    model = pickle.load(file)

# Set up the Streamlit interface
st.title('Credit Card Fraud Detection')
st.write('Enter the transaction details below:')

# Input fields for transaction data
transaction_amount = st.number_input('Transaction Amount', min_value=0.0, step=0.01)
# Add additional input fields for other features as needed

# Process input data
data = {
    'amount': transaction_amount,
    # Add other features
}
input_df = pd.DataFrame([data])

# Predict
if st.button('Predict'):
    # Standardize or preprocess the input data if needed
    prediction = model.predict(input_df)
    st.write('Prediction:', 'Fraudulent' if prediction[0] == 1 else 'Legitimate')
    
**Running the Streamlit App**

Install Streamlit if not already installed:


pip install streamlit

**Run the Streamlit app:**

streamlit run streamlit(credit_card)(1)(1).py


import streamlit as st
import pandas as pd
import pickle
from sklearn.preprocessing import StandardScaler, PowerTransformer

# Function to load the model from a specified file path
def load_model(file_path):
    with open(file_path, 'rb') as file:
        model = pickle.load(file)
    return model

# Path to the model file
model_path = 'load_best_model.pkl'  # replace with the correct path to your best_model.pkl

# Load the model
model = load_model(model_path)
st.success("Model loaded successfully")

# Title of the app
st.write("""
# Credit Card Fraud Detection
This app predicts **Credit Card Fraud** based on transaction data!
""")

# Sidebar for user input features
st.sidebar.header('User Input Parameters')

def user_input_features():
    V1 = st.sidebar.slider('V1', -56.407510, 2.454930, 1.759061e-12)
    V2 = st.sidebar.slider('V2', -72.715728, 22.057729, -8.251130e-13)
    V3 = st.sidebar.slider('V3', -48.325589, 9.382558, -9.654937e-13)
    V4 = st.sidebar.slider('V4', -5.683171, 16.875344, 8.321385e-13)
    V5 = st.sidebar.slider('V5', -113.743307, 34.801666, 1.649999e-13)
    V6 = st.sidebar.slider('V6', -26.160506, 73.301626, 4.248366e-13)
    V7 = st.sidebar.slider('V7', -43.557242, 120.589494, -3.054600e-13)
    V8 = st.sidebar.slider('V8', -73.216718, 20.007208, 8.777971e-14)
    V9 = st.sidebar.slider('V9', -13.434066, 15.594995, -1.179749e-12)
    V10 = st.sidebar.slider('V10', -24.588262, 23.745136, 7.092545e-13)
    V11 = st.sidebar.slider('V11', -4.797473, 12.018913, 1.874948e-12)
    V12 = st.sidebar.slider('V12', -18.683715, 7.848392, 1.053347e-12)
    V13 = st.sidebar.slider('V13', -5.791881, 7.126883, 7.127611e-13)
    V14 = st.sidebar.slider('V14', -19.214325, 10.526766, -1.474791e-13)
    V15 = st.sidebar.slider('V15', -4.498945, 8.877742, -5.231558e-13)
    V16 = st.sidebar.slider('V16', -14.129855, 17.315112, -2.282250e-13)
    V17 = st.sidebar.slider('V17', -25.162799, 9.253526, -6.425436e-13)
    V18 = st.sidebar.slider('V18', -9.498746, 5.041069, 4.950748e-13)
    V19 = st.sidebar.slider('V19', -7.213527, 5.591971, 7.057397e-13)
    V20 = st.sidebar.slider('V20', -54.497720, 39.420904, 1.766111e-12)
    V21 = st.sidebar.slider('V21', -34.830382, 27.202839, -3.405756e-13)
    V22 = st.sidebar.slider('V22', -10.933144, 10.503090, -5.723197e-13)
    V23 = st.sidebar.slider('V23', -44.807735, 22.528412, -9.725856e-13)
    V24 = st.sidebar.slider('V24', -2.836627, 4.584549, 1.464150e-12)
    V25 = st.sidebar.slider('V25', -10.295397, 7.519589, -6.987102e-13)
    V26 = st.sidebar.slider('V26', -2.604551, 3.517346, -5.617874e-13)
    V27 = st.sidebar.slider('V27', -22.565679, 31.612198, 3.332082e-12)
    V28 = st.sidebar.slider('V28', -15.430084, 33.847808, -3.518874e-12)
    Amount = st.sidebar.slider('Amount', 0.000000, 25691.160000, 88.34962)
    
    data = {
        'V1': V1, 'V2': V2, 'V3': V3, 'V4': V4, 'V5': V5, 'V6': V6, 'V7': V7, 
        'V8': V8, 'V9': V9, 'V10': V10, 'V11': V11, 'V12': V12, 'V13': V13, 
        'V14': V14, 'V15': V15, 'V16': V16, 'V17': V17, 'V18': V18, 'V19': V19, 
        'V20': V20, 'V21': V21, 'V22': V22, 'V23': V23, 'V24': V24, 'V25': V25, 
        'V26': V26, 'V27': V27, 'V28': V28, 'Amount': Amount
    }
    features = pd.DataFrame(data, index=[0])
    
    return features

# Preprocessing function
def preprocess_data(df):
    # Assuming the scaler and power transformer were fit on the original training data
    scaler = StandardScaler()
    power_transformer = PowerTransformer(method='yeo-johnson', standardize=True)
    
    try:
        # Apply power transformation and scaling only to the Amount column
        df['Amount'] = power_transformer.fit_transform(df[['Amount']])
        df['Amount'] = scaler.fit_transform(df[['Amount']])
    except ValueError as e:
        st.error(f"Error in preprocessing data: {e}")
        return df

    return df

# Get user input features
df = user_input_features()

st.subheader('User Input Parameters')
st.write(df)

# Preprocess the user input features
preprocessed_data = preprocess_data(df)

# Make predictions
if model:
    prediction = model.predict(preprocessed_data)
    prediction_proba = model.predict_proba(preprocessed_data)

    st.subheader('Prediction')
    if prediction == 1:
        st.write('Fraudulent transaction')
    else:
        st.write('Not a fraudulent transaction')

    st.subheader('Prediction Probability')
    st.write(prediction_proba)
else:
    st.error("Model not loaded.")




## Conclusion
Project Summary:
The FindDefault project aimed to develop a reliable credit card fraud detection model, addressing the critical challenge of identifying fraudulent transactions among a vast number of legitimate ones. The project followed a systematic approach, ensuring that each phase contributed to building a robust and accurate predictive model.

1.	Exploratory Data Analysis (EDA):
o	The project began with a thorough exploration of the dataset, where key insights into the distribution and characteristics of the data were uncovered. This step was crucial for identifying potential challenges, such as the significant class imbalance, and understanding the relationships between various features.
2.	Data Cleaning and Preprocessing:
o	Data quality was ensured by addressing any missing values, outliers, and inconsistencies in the dataset. Numerical features were standardized, and necessary transformations were applied to ensure the data was in an optimal state for modeling.
3.	Handling Imbalanced Data:
o	Given the substantial imbalance between legitimate and fraudulent transactions, several techniques, including SMOTE and ADASYN, were employed to balance the dataset. This step was critical in preventing the model from becoming biased towards the majority class.
4.	Feature Engineering:
o	New features were engineered to capture additional information that was not directly available in the original dataset. Dimensionality reduction techniques, such as PCA, were applied where necessary to streamline the feature space and enhance model performance.
5.	Model Selection and Training:
o	Multiple machine learning models, including Logistic Regression, Decision Trees, and XGBoost, were considered and trained. Hyperparameter tuning was performed using GridSearchCV to identify the best model configurations, ensuring optimal performance.
6.	Model Evaluation:
o	The models were rigorously evaluated using various metrics, including Accuracy, Precision, Recall, F1-Score, and ROC-AUC. The Logistic Regression model, balanced with SMOTE, was selected as the final model due to its superior performance, achieving a high ROC-AUC score on both the training and test datasets.
7.	Model Deployment:
o	The final model was serialized using the pickle module, making it ready for deployment in a production environment. This deployment allows the model to make real-time predictions on new transactions, providing a valuable tool for detecting credit card fraud.



**Documentation Benefits to the Company**

The detailed documentation provided for the FindDefault project exemplifies a commitment to thoroughness and clarity, which directly benefits the company in several ways:
1.	Improved Collaboration:
o	Comprehensive documentation ensures that all team members, regardless of their role, have a clear understanding of the project’s objectives, methodologies, and outcomes. This facilitates better collaboration and more efficient problem-solving.
2.	Streamlined Onboarding:
o	New team members can quickly get up to speed by reviewing the documentation, reducing the time needed for onboarding and allowing them to contribute effectively from the outset.
3.	Enhanced Decision-Making:
o	Clear documentation of model selection, evaluation criteria, and results allows stakeholders to make informed decisions based on the project’s findings. This transparency is crucial for building trust in the model's predictions and the overall project outcomes.
4.	Regulatory Compliance:
o	In highly regulated industries like finance, having detailed documentation is essential for demonstrating compliance with industry standards and regulations. It provides a record of the methodologies used and the reasoning behind key decisions.
5.	Future-Proofing:
o	Detailed documentation lays the foundation for future work, making it easier to revisit the project, iterate on the existing model, or integrate new technologies as they become available. This ensures that the project remains relevant and continues to provide value over time.
