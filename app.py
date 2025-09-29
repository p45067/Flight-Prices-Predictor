import streamlit as st
import pandas as pd
import joblib

# Load trained model
model = joblib.load("xgboost_flight_pipeline.pkl")

# Streamlit UI
st.set_page_config(page_title="Flight Price Predictor", page_icon="✈️")
st.title("✈️ Flight Price Prediction App")
st.write("Predict the price of a flight ticket based on travel details.")

# Input fields
airline = st.selectbox("Airline", [
    "SpiceJet", "AirAsia", "Vistara", "Indigo", "GoAir", "Air_India"
])

source_city = st.selectbox("Source City", [
    "Delhi", "Mumbai", "Bangalore", "Kolkata", "Chennai", "Hyderabad"
])

destination_city = st.selectbox("Destination City", [
    "Delhi", "Mumbai", "Bangalore", "Kolkata", "Chennai", "Hyderabad"
])

departure_time = st.selectbox("Departure Time", [
    "Early_Morning", "Morning", "Afternoon", "Evening", "Night", "Late_Night"
])

arrival_time = st.selectbox("Arrival Time", [
    "Early_Morning", "Morning", "Afternoon", "Evening", "Night", "Late_Night"
])

stops = st.selectbox("Number of Stops", [
    "zero", "one", "two_or_more"
])

travel_class = st.selectbox("Class", ["Economy", "Business"])

duration = st.number_input("Duration (hours)", min_value=0.5, max_value=20.0, step=0.1)
days_left = st.number_input("Days Left for Departure", min_value=0, max_value=365, step=1)

# Prepare input dataframe
input_data = pd.DataFrame([{
    "airline": airline,
    "source_city": source_city,
    "departure_time": departure_time,
    "stops": stops,
    "arrival_time": arrival_time,
    "destination_city": destination_city,
    "class": travel_class,
    "duration": duration,
    "days_left": days_left
}])

# Predict button
if st.button("Predict Price"):
    prediction = model.predict(input_data)[0]
    st.success(f"💰 Estimated Flight Price: ₹{round(prediction, 2)}")
