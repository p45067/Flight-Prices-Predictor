import streamlit as st
import pandas as pd
import joblib

# Load scaler and models
scaler = joblib.load("scaler.joblib")
logistic_model = joblib.load("logistic_model.joblib")
svc_model = joblib.load("svc_model.joblib")

st.title("🚚 Delivery Delay Prediction App")
st.write("Predict whether a delivery will be delayed based on operational factors.")

# Sidebar model selection
model_choice = st.sidebar.selectbox("Select Model", ("Logistic Regression", "SVC"))

# User inputs
st.header("Enter Delivery Details")

delivery_distance = st.number_input("Delivery Distance (km)", min_value=0.0, step=0.1)
driver_experience = st.number_input("Driver Experience (years)", min_value=0, step=1)
num_stops = st.number_input("Number of Stops", min_value=0, step=1)
vehicle_age = st.number_input("Vehicle Age (years)", min_value=0, step=1)
road_condition = st.slider("Road Condition Score (1=Poor, 5=Excellent)", 1, 5, 3)
package_weight = st.number_input("Package Weight (kg)", min_value=0.0, step=0.1)
fuel_efficiency = st.number_input("Fuel Efficiency (km/l)", min_value=0.0, step=0.1)
processing_time = st.number_input("Warehouse Processing Time (minutes)", min_value=0, step=1)

traffic_congestion = st.selectbox("Traffic Congestion", ["Low", "Medium", "High"])
weather_condition = st.selectbox("Weather Condition", ["Clear", "Rainy", "Stormy", "Foggy"])
delivery_slot = st.selectbox("Delivery Slot", ["Morning", "Afternoon", "Evening", "Night"])

# Prepare input dictionary
input_dict = {
    "Delivery Distance": delivery_distance,
    "Driver Experience": driver_experience,
    "Number of Stops": num_stops,
    "Vehicle Age": vehicle_age,
    "Road Condition Score": road_condition,
    "Package Weight": package_weight,
    "Fuel Efficiency": fuel_efficiency,
    "Warehouse Processing Time": processing_time,
    "Traffic Congestion_Medium": 1 if traffic_congestion == "Medium" else 0,
    "Traffic Congestion_High": 1 if traffic_congestion == "High" else 0,
    "Weather Condition_Rainy": 1 if weather_condition == "Rainy" else 0,
    "Weather Condition_Stormy": 1 if weather_condition == "Stormy" else 0,
    "Weather Condition_Foggy": 1 if weather_condition == "Foggy" else 0,
    "Delivery Slot_Afternoon": 1 if delivery_slot == "Afternoon" else 0,
    "Delivery Slot_Evening": 1 if delivery_slot == "Evening" else 0,
    "Delivery Slot_Night": 1 if delivery_slot == "Night" else 0
}

# Convert input to DataFrame
input_df = pd.DataFrame([input_dict])

# Scale numeric + dummy features
X_scaled = scaler.transform(input_df)

# Prediction
if st.button("Predict Delay"):
    if model_choice == "Logistic Regression":
        prob = logistic_model.predict_proba(X_scaled)[0][1]
        pred = logistic_model.predict(X_scaled)[0]
    else:
        prob = svc_model.predict_proba(X_scaled)[0][1]
        pred = svc_model.predict(X_scaled)[0]
    
    st.write(f"**Prediction:** {'Delayed' if pred == 1 else 'On Time'}")
    st.write(f"**Probability of Delay:** {prob:.2f}")
