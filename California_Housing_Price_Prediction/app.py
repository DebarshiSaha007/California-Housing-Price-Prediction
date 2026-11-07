
import streamlit as st
import pandas as pd
import joblib

# Load trained model and training feature ranges
model = joblib.load("california_housing_model.pkl")
feature_ranges = joblib.load("housing_feature_ranges.pkl")

st.set_page_config(
    page_title="California Housing Price Predictor",
    page_icon="🏠",
    layout="centered"
)

st.title("🏠 California Housing Price Prediction")

# ABOUT THE PROJECT
st.header("About This Application")

st.write("""
This application estimates the median house value in a California
census block using a Machine Learning model.

It uses a Random Forest Regressor trained on the California Housing
dataset available through Scikit-learn. The dataset is based on
the 1990 U.S. Census.

The model learns relationships between housing characteristics
and median house values from historical data. When you enter
housing details, it processes those values and predicts an
estimated median house value.
""")

st.info("""
**Important dataset limitation:** The training data represents
California housing conditions around 1990. This application is
an educational machine learning demonstration, not a source of
current property prices or professional property valuations.
""")

# HOW IT WORKS
st.header("How Does It Work?")

st.write("""
1. Enter the eight housing-related features.
2. The application sends these values to the trained Random Forest model.
3. The model combines predictions from multiple decision trees.
4. The application displays the estimated median house value.
5. An Input Coverage Score indicates how closely your inputs
   fall within the ranges represented in the training data.
""")

# INPUT GUIDE
st.header("Understanding the Input Features")

st.write("""
- **Median Income:** Median household income in the census block,
  measured in tens of thousands of dollars.
- **House Age:** Median age of houses in the block, in years.
- **Average Rooms:** Average number of rooms per household.
- **Average Bedrooms:** Average number of bedrooms per household.
- **Population:** Total population of the census block.
- **Average Occupancy:** Average number of people per household.
- **Latitude:** Geographic latitude of the area.
- **Longitude:** Geographic longitude of the area.
""")

st.caption(
    "Decimal precision: Inputs are accepted up to 2 decimal places. "
    "Values are displayed with two decimal places."
)

st.divider()

# INPUT FIELDS
st.header("Enter Housing Details")

med_inc = st.number_input(
    "Median Income (in $10,000 units)",
    min_value=0.0,
    value=5.0,
    step=0.01,
    format="%.2f"
)

house_age = st.number_input(
    "House Age (years)",
    min_value=0.0,
    value=20.0,
    step=0.01,
    format="%.2f"
)

ave_rooms = st.number_input(
    "Average Rooms",
    min_value=0.0,
    value=5.0,
    step=0.01,
    format="%.2f"
)

ave_bedrms = st.number_input(
    "Average Bedrooms",
    min_value=0.0,
    value=1.0,
    step=0.01,
    format="%.2f"
)

population = st.number_input(
    "Population",
    min_value=0.0,
    value=1000.0,
    step=0.01,
    format="%.2f"
)

ave_occup = st.number_input(
    "Average Occupancy",
    min_value=0.0,
    value=3.0,
    step=0.01,
    format="%.2f"
)

latitude = st.number_input(
    "Latitude",
    min_value=32.0,
    max_value=42.0,
    value=34.0,
    step=0.01,
    format="%.2f"
)

longitude = st.number_input(
    "Longitude",
    min_value=-125.0,
    max_value=-114.0,
    value=-118.0,
    step=0.01,
    format="%.2f"
)

st.divider()

# PREDICTION
if st.button("Predict House Price", type="primary"):

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

    prediction = model.predict(input_data)[0]
    estimated_price = prediction * 100000

    st.success(
        f"Estimated Median House Value: ${estimated_price:,.2f}"
    )

    # INPUT COVERAGE SCORE
    st.header("Input Coverage Score")

    lower_limits = feature_ranges.loc[0.01]
    upper_limits = feature_ranges.loc[0.99]

    in_range_count = 0

    for feature in input_data.columns:
        value = input_data.iloc[0][feature]

        if lower_limits[feature] <= value <= upper_limits[feature]:
            in_range_count += 1

    coverage_score = (in_range_count / len(input_data.columns)) * 100

    st.metric(
        "Input Coverage",
        f"{coverage_score:.0f}%",
        f"{in_range_count} of 8 features within range"
    )

    st.progress(int(coverage_score))

    st.write("""
    **What does this score mean?**

    The Input Coverage Score measures the percentage of input
    features that fall between the 1st and 99th percentiles
    of the corresponding training feature distributions.

    A higher score means more of your inputs fall within
    the ranges commonly represented in the training data.
    """)

    st.warning("""
    **Important:** This is NOT a statistical confidence score
    or a probability that the prediction is correct. It does
    not measure prediction error or guarantee accuracy.
    It only indicates how well the input values are covered
    by the training data ranges.
    """)

    st.caption(
        "All predictions are historical-data-based estimates, "
        "not live property valuations."
    )
