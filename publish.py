import streamlit as st
import pandas as pd
import xgboost as xgb
import numpy as np
from xgboost import XGBRegressor

best_model = XGBRegressor()
best_model.load_model('best_model.json')

# Define the app layout
st.title('Crop Yield Prediction App')

# Create input fields for the features
rain_fall = st.number_input('Rain Fall (mm)', value=1000.0)
fertilizer = st.number_input('Fertilizer', value=70.0)
temperature = st.number_input('Temperatue', value=30.0)
nitrogen = st.number_input('Nitrogen (N)', value=75.0)
phosphorus = st.number_input('Phosphorus (P)', value=20.0)
potassium = st.number_input('Potassium (K)', value=20.0)

# Button to make prediction
if st.button('Predict Yield'):
    # Prepare the feature vector for prediction
    new_data = pd.DataFrame({
        'Rain Fall (mm)': [rain_fall],
        'Fertilizer': [fertilizer],
        'Temperatue': [temperature],
        'Nitrogen (N)': [nitrogen],
        'Phosphorus (P)': [phosphorus],
        'Potassium (K)': [potassium]
    })
    
    # Predict the crop yield
    prediction = best_model.predict(new_data)[0]
    
    # Display the prediction
    st.write(f'Predicted Crop Yield: {prediction} Q/acre')
