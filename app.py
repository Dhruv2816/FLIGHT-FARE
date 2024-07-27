import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# Load models and preprocessor
MODEL_PATH = r'C:\Users\dream\OneDrive\Desktop\Flight_Flare_New\artifacts\xgboost_model.pkl'
PREPROCESSOR_PATH = r'C:\Users\dream\OneDrive\Desktop\Flight_Flare_New\artifacts\preprocessor.pkl'
COLUMN_NAMES_PATH = r'C:\Users\dream\OneDrive\Desktop\Flight_Flare_New\artifacts\column_names.pkl'

@st.cache(allow_output_mutation=True)
def load_model():
    model = joblib.load(MODEL_PATH)
    return model

@st.cache(allow_output_mutation=True)
def load_preprocessor():
    preprocessor = joblib.load(PREPROCESSOR_PATH)
    return preprocessor

@st.cache(allow_output_mutation=True)
def load_column_names():
    with open(COLUMN_NAMES_PATH, 'rb') as file:
        column_names = joblib.load(file)
    return column_names

model = load_model()
preprocessor = load_preprocessor()
column_names = load_column_names()

# App title
st.title("Flight Fare Prediction")

# Input form
st.header("Enter Flight Details")
airline = st.selectbox("Airline", options=['AirIndia', 'Airline2', 'Other'])  # Update options as needed
source = st.selectbox("Source", options=['Source1', 'Source2', 'Other'])      # Update options as needed
destination = st.selectbox("Destination", options=['Destination1', 'Destination2', 'Other'])  # Update options as needed
journey_day = st.selectbox("Journey Day", options=['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'])
departure = st.selectbox("Departure Time", options=['Before 6 AM', '6 AM - 12 PM', '12 PM - 6 PM', 'After 6 PM'])
arrival = st.selectbox("Arrival Time", options=['Before 6 AM', '6 AM - 12 PM', '12 PM - 6 PM', 'After 6 PM'])
total_stops = st.selectbox("Total Stops", options=['Non-stop', '1 Stop', '2 Stops', '3 Stops'])
duration_in_hours = st.number_input("Duration (in hours)", min_value=0.0, max_value=24.0, step=0.1)
days_left = st.number_input("Days Left to Departure", min_value=0, max_value=365, step=1)
class_type = st.selectbox("Class", options=['Economy', 'Business'])

# Predict button
if st.button("Predict Fare"):
    # Prepare input data
    input_data = pd.DataFrame({
        'Airline': [airline],
        'Source': [source],
        'Destination': [destination],
        'Journey_day': [journey_day],
        'Departure': [departure],
        'Arrival': [arrival],
        'Total_stops': [total_stops],
        'Duration_in_hours': [duration_in_hours],
        'Days_left': [days_left],
        'Class': [class_type],
    })

    # Feature engineering
    input_data['Journey_month'] = pd.to_datetime('today').month
    input_data['On_weekend'] = input_data['Journey_day'].isin(['Saturday', 'Sunday'])
    input_data['Daytime_departure'] = ~input_data['Departure'].isin(['After 6 PM', 'Before 6 AM'])
    input_data['Daytime_arrival'] = ~input_data['Arrival'].isin(['After 6 PM', 'Before 6 AM'])

    # Reorder columns to match training data
    input_data = input_data[column_names['features']]

    # Preprocess input data
    input_features = preprocessor.transform(input_data)

    # Make prediction
    predicted_fare = model.predict(input_features)
    st.success(f"Estimated Flight Fare: ₹{predicted_fare[0]:.2f}")
