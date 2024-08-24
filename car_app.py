
import streamlit as st
import pandas as pd
import numpy as np
import pickle

# Load the retrained model and the model columns
model = pickle.load(open('best_rf_model.pkl', 'rb'))
model_columns = pickle.load(open('model_columns.pkl', 'rb'))
feature_scaler = pickle.load(open('feature_scaler.pkl', 'rb'))  # Load the feature scaler
target_scaler = pickle.load(open('target_scaler.pkl', 'rb'))    # Load the target scaler

# Define the Streamlit app
st.title("Car Price Prediction App")

# Collecting user input for the features
st.header("Input the Details of the Car")

# Collecting user inputs
brand = st.selectbox("Select the Brand", ["Maruti", "Hyundai", "Mahindra", "Toyota", "Honda", "Ford", "Renault", "Tata", "Chevrolet", "Volkswagen", "Nissan", "Skoda", "Mercedes", "BMW", "Audi", "Jaguar", "Other"])
seller_type = st.selectbox("Seller Type", ["Dealer", "Individual", "Trustmark Dealer"])
transmission = st.selectbox("Transmission Type", ["Manual", "Automatic"])
owner = st.selectbox("Owner Type", ["First Owner", "Second Owner", "Third Owner", "Fourth & Above Owner", "Test Drive Car"])
fuel = st.selectbox("Fuel Type", ["Petrol", "Diesel", "CNG", "LPG", "Electric"])
year = st.slider("Year of Manufacture", min_value=1990, max_value=2024, value=2010)
km_driven = st.slider("Kilometers Driven", min_value=0, max_value=500000, value=30000, step=1000)

# Prepare the input data as a DataFrame
input_data = pd.DataFrame({
    'brand': [brand],
    'year': [year],
    'km_driven': [km_driven],
    'seller_type': [seller_type],
    'transmission': [transmission],
    'owner': [owner],
    'fuel': [fuel]
})

# One-Hot Encoding for categorical variables
input_data_encoded = pd.get_dummies(input_data)

# Ensure the input data matches the columns used during model training
input_data_encoded = input_data_encoded.reindex(columns=model_columns, fill_value=0)

# Scale the feature data
input_data_encoded[['km_driven']] = feature_scaler.transform(input_data_encoded[['km_driven']])

# Predict the selling price using the model
prediction_scaled = model.predict(input_data_encoded)

# Reverse scaling of the predicted value
prediction_unscaled = target_scaler.inverse_transform(prediction_scaled.reshape(-1, 1))

# Display the prediction
st.subheader("Predicted Selling Price")
st.write(f"The estimated selling price of the car is: ₹ {prediction_unscaled[0, 0]:,.2f}")



