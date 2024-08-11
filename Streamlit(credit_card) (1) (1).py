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

