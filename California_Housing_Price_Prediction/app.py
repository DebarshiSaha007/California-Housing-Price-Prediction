import streamlit as st
import pandas as pd
import joblib

# Load the trained Random Forest model
model = joblib.load("california_housing_model.pkl")

# Configure the Streamlit page
st.set_page_config(
    page_title="California House Price Predictor",
    page_icon="🏠",
    layout="centered"
)

# App heading
st.title("🏠 California Housing Price Prediction")

st.write(
    "Enter the housing details below to estimate "
    "the median house value in California."
)

st.divider()

# Housing input section
st.subheader("Enter Housing Details")

med_inc = st.number_input(
    "Median Income",
    min_value=0.0,
    value=5.0
)

house_age = st.number_input(
    "House Age",
    min_value=0.0,
    value=20.0
)

ave_rooms = st.number_input(
    "Average Rooms",
    min_value=0.0,
    value=5.0
)

ave_bedrms = st.number_input(
    "Average Bedrooms",
    min_value=0.0,
    value=1.0
)

population = st.number_input(
    "Population",
    min_value=0.0,
    value=1000.0
)

ave_occup = st.number_input(
    "Average Occupancy",
    min_value=0.0,
    value=3.0
)

latitude = st.number_input(
    "Latitude",
    min_value=32.0,
    max_value=42.0,
    value=34.0
)

longitude = st.number_input(
    "Longitude",
    min_value=-125.0,
    max_value=-114.0,
    value=-118.0
)

st.divider()

# Prediction button
if st.button("Predict House Price"):

    # Prepare input with the same feature names
    # and order used during model training
    input_data = pd.DataFrame(
        [[
            med_inc,
            house_age,
            ave_rooms,
            ave_bedrms,
            population,
            ave_occup,
            latitude,
            longitude
        ]],
        columns=[
            "MedInc",
            "HouseAge",
            "AveRooms",
            "AveBedrms",
            "Population",
            "AveOccup",
            "Latitude",
            "Longitude"
        ]
    )

    # Predict house value
    prediction = model.predict(input_data)[0]

    # Convert target units ($100,000) into dollars
    estimated_price = prediction * 100000

    # Display prediction
    st.success(
        f"Estimated Median House Value: "
        f"${estimated_price:,.2f}"
    )

    st.caption(
        "This is a machine learning estimate, "
        "not a live property valuation."
    )