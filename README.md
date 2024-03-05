# crop_yield_pred
predicting crop yield based on several variable

# Necessary imports
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import xgboost as xgb

# Correct the file path to include the Excel file name
file_path = '/content/drive/MyDrive/crop yield data sheet.xlsx'
data = pd.read_excel(file_path)
data['Temperatue'] = pd.to_numeric(data['Temperatue'], errors='coerce')
data.fillna(data.mean(), inplace=True)

# Split the dataset
X = data.drop('Yeild (Q/acre)', axis=1)
y = data['Yeild (Q/acre)']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize XGBoost regressor
xgb_model = xgb.XGBRegressor(objective='reg:squarederror', random_state=42)

# Define the parameter grid for hyperparameter tuning
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [3, 4, 5],
    'learning_rate': [0.01, 0.05, 0.1],
    'subsample': [0.8, 0.9, 1.0]
}

# Setup GridSearchCV
grid_search = GridSearchCV(estimator=xgb_model, param_grid=param_grid, cv=3, n_jobs=-1, verbose=2, scoring='r2')

# Perform hyperparameter tuning
grid_search.fit(X_train, y_train)

# Best model after grid search
best_model = grid_search.best_estimator_

# Predict on the testing set
y_pred = best_model.predict(X_test)

# Calculate performance metrics
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f'MAE: {mae}')
print(f'MSE: {mse}')
print(f'R² Score: {r2}')

import streamlit as st
import pandas as pd
import xgboost as xgb
import numpy as np

# Load your trained model
# Ensure the model is loaded correctly, e.g.,
# best_model = xgb.XGBRegressor(...)
# best_model.load_model('path_to_your_model_file')

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
