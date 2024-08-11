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

Hyperparameter Tuning
To enhance model performance, hyperparameters were meticulously tuned using GridSearchCV. This technique involves systematically exploring a specified range of hyperparameters for each model to identify the optimal combination that maximizes performance metrics. GridSearchCV evaluates multiple parameter configurations by performing cross-validation, ensuring the selected hyperparameters provide the best trade-off between model accuracy and generalization.

Choosing the Best Model
After tuning hyperparameters, the performance of various models was assessed, and the Logistic Regression model emerged as the top performer. This model, trained on a dataset balanced using SMOTE, demonstrated outstanding performance with an ROC-AUC score of 0.99 on the training set and 0.97 on the test set. These high scores indicate the model’s excellent ability to differentiate between fraudulent and legitimate transactions.

The decision to select Logistic Regression was based on several factors:

Simplicity: The model’s straightforward nature makes it easy to interpret and understand.
Ease of Interpretation: Logistic Regression provides clear insights into the influence of each feature on the prediction, aiding in interpretability.
Lower Computational Resource Requirements: Compared to more complex models like XGBoost, Logistic Regression is computationally efficient, making it suitable for deployment in real-time systems.
By combining hyperparameter tuning with the selected optimal model, the project ensures that the final Logistic Regression model is both highly accurate and practical for real-world application.


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


**Future Work**

1. Integration of Real-Time Data Streams
Description: Implementing a real-time data ingestion and processing pipeline to handle streaming credit card transaction data. This could involve integrating with tools like Apache Kafka or Spark Streaming to process transactions in real-time, allowing for immediate fraud detection.
Benefit: Enhances the system’s capability to detect and respond to fraudulent transactions as they happen, reducing financial loss and increasing the system's practical value.
2. Exploration of Deep Learning Models
Description: Investigating the application of deep learning techniques such as Recurrent Neural Networks (RNNs) or Long Short-Term Memory (LSTM) networks, which could better capture temporal patterns and dependencies in transaction data.
Benefit: These models might uncover complex patterns in the data that traditional machine learning models cannot, potentially improving fraud detection accuracy.
3. Incorporating Additional Data Sources
Description: Expanding the dataset by incorporating external data sources, such as customer demographics, spending behavior patterns, or transaction metadata (e.g., device type, location data).
Benefit: Additional contextual information could improve the model’s ability to differentiate between fraudulent and legitimate transactions, especially in edge cases.
4. Model Explainability and Transparency
Description: Implementing techniques like SHAP (SHapley Additive exPlanations) or LIME (Local Interpretable Model-agnostic Explanations) to make the model’s decisions more interpretable to stakeholders.
Benefit: Enhances trust and adoption of the model by providing insights into how and why certain predictions are made, which is particularly important in regulated industries.
5. Deployment of an AutoML Framework
Description: Exploring the use of Automated Machine Learning (AutoML) frameworks to automate model selection, hyperparameter tuning, and feature engineering. This could include tools like Google AutoML, H2O.ai, or Azure AutoML.
Benefit: Reduces the time and effort required for model development, while potentially uncovering better-performing models through automated processes.
6. Implementation of Anomaly Detection Techniques
Description: Integrating unsupervised learning methods, such as Isolation Forests or Autoencoders, to detect anomalous transactions that may not have been labeled in the training data.
Benefit: Improves the detection of novel or rare types of fraud that the supervised models may not have been trained to recognize.
7. Periodic Model Retraining and Monitoring
Description: Setting up a system for continuous model monitoring and periodic retraining to adapt to changes in transaction patterns or emerging fraud tactics.
Benefit: Ensures that the model remains accurate and effective over time, especially as fraudsters evolve their strategies.
