"""
House Price Prediction — Interactive Dashboard
================================================
Run with:
    streamlit run dashboard.py

Requires train_model.py to have been run first (produces best_model.pkl,
feature_cols.pkl, housing_data.csv, test_data.csv, model_metadata.json).
"""

import json
import os

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(page_title="House Price Predictor", page_icon="🏠", layout="wide")

ARTIFACTS = ["best_model.pkl", "feature_cols.pkl", "housing_data.csv", "test_data.csv", "model_metadata.json"]


@st.cache_resource
def load_artifacts():
    missing = [f for f in ARTIFACTS if not os.path.exists(f)]
    if missing:
        return None
    model = joblib.load("best_model.pkl")
    feature_cols = joblib.load("feature_cols.pkl")
    df = pd.read_csv("housing_data.csv")
    test_df = pd.read_csv("test_data.csv")
    with open("model_metadata.json") as f:
        metadata = json.load(f)
    return model, feature_cols, df, test_df, metadata


artifacts = load_artifacts()

if artifacts is None:
    st.error(
        "Model artifacts not found. Please run `python train_model.py` first "
        "to train the model and generate the required files."
    )
    st.stop()

model, feature_cols, df, test_df, metadata = artifacts

st.title("🏠 House Price Prediction Dashboard")
st.caption(f"Best model in use: **{metadata['best_model']}**")

tab1, tab2, tab3 = st.tabs(["📊 Data Overview", "📈 Model Performance", "🔮 Predict a Price"])

# ----------------------------------------------------------------------------
# TAB 1 — Data Overview
# ----------------------------------------------------------------------------
with tab1:
    st.subheader("Dataset Summary")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Records", f"{len(df):,}")
    c2.metric("Median Price", f"${df['MedHouseVal'].median():,.0f}")
    c3.metric("Min Price", f"${df['MedHouseVal'].min():,.0f}")
    c4.metric("Max Price", f"${df['MedHouseVal'].max():,.0f}")

    col1, col2 = st.columns(2)
    with col1:
        fig = px.histogram(df, x="MedHouseVal", nbins=50, title="Price Distribution")
        fig.update_layout(xaxis_title="Price ($)", yaxis_title="Count")
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        fig = px.scatter(
            df, x="SqFtLiving", y="MedHouseVal", color="OverallQuality",
            title="Price vs Living Area (colored by Quality)",
            opacity=0.5, color_continuous_scale="Viridis",
        )
        fig.update_layout(xaxis_title="Sq Ft Living", yaxis_title="Price ($)")
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Feature Correlations with Price")
    numeric_df = df.select_dtypes(include=[np.number])
    corr = numeric_df.corr()["MedHouseVal"].drop("MedHouseVal").sort_values()
    fig = px.bar(
        x=corr.values, y=corr.index, orientation="h",
        title="Correlation with House Price",
        labels={"x": "Correlation", "y": "Feature"},
        color=corr.values, color_continuous_scale="RdBu", color_continuous_midpoint=0,
    )
    st.plotly_chart(fig, use_container_width=True)

    with st.expander("View raw data sample"):
        st.dataframe(df.sample(min(200, len(df)), random_state=1), use_container_width=True)

# ----------------------------------------------------------------------------
# TAB 2 — Model Performance
# ----------------------------------------------------------------------------
with tab2:
    st.subheader("Model Comparison")
    results_df = pd.DataFrame(metadata["results"]).T
    results_df = results_df.sort_values("RMSE")
    st.dataframe(
        results_df.style.format({"RMSE": "${:,.0f}", "MAE": "${:,.0f}", "R2": "{:.4f}"})
        .highlight_min(subset=["RMSE", "MAE"], color="#d4f7d4")
        .highlight_max(subset=["R2"], color="#d4f7d4"),
        use_container_width=True,
    )

    col1, col2 = st.columns(2)
    with col1:
        fig = px.bar(
            results_df.reset_index(), x="index", y="RMSE",
            title="RMSE by Model (lower is better)", labels={"index": "Model"},
        )
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        fig = px.bar(
            results_df.reset_index(), x="index", y="R2",
            title="R² by Model (higher is better)", labels={"index": "Model"},
        )
        st.plotly_chart(fig, use_container_width=True)

    st.subheader(f"Predicted vs Actual — {metadata['best_model']}")
    X_test = test_df[feature_cols]
    y_test = test_df["MedHouseVal"]
    preds = model.predict(X_test)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=y_test, y=preds, mode="markers", opacity=0.4, name="Predictions"))
    lims = [min(y_test.min(), preds.min()), max(y_test.max(), preds.max())]
    fig.add_trace(go.Scatter(x=lims, y=lims, mode="lines", name="Perfect Prediction", line=dict(dash="dash")))
    fig.update_layout(xaxis_title="Actual Price ($)", yaxis_title="Predicted Price ($)", height=500)
    st.plotly_chart(fig, use_container_width=True)

    if metadata.get("feature_importance"):
        st.subheader("Feature Importance")
        imp = pd.Series(metadata["feature_importance"]).sort_values()
        fig = px.bar(x=imp.values, y=imp.index, orientation="h", labels={"x": "Importance", "y": "Feature"})
        st.plotly_chart(fig, use_container_width=True)

# ----------------------------------------------------------------------------
# TAB 3 — Predict a Price
# ----------------------------------------------------------------------------
with tab3:
    st.subheader("Enter House Details")
    st.caption("Adjust the sliders/inputs to get a live price prediction.")

    d1, d2, d3 = st.columns(3)
    with d1:
        sqft_living = st.slider("Living Area (sq ft)", 400, 6000, 1800, step=50)
        bedrooms = st.slider("Bedrooms", 1, 6, 3)
        bathrooms = st.slider("Bathrooms", 1.0, 5.0, 2.0, step=0.5)
        lot_size = st.slider("Lot Size (sq ft)", 1000, 40000, 8000, step=500)
    with d2:
        year_built = st.slider("Year Built", 1920, 2023, 2000)
        overall_quality = st.slider("Overall Quality (1-10)", 1, 10, 6)
        garage_cars = st.slider("Garage Capacity (cars)", 0, 4, 2)
        stories = st.selectbox("Stories", [1, 1.5, 2, 2.5, 3], index=2)
    with d3:
        distance_to_city_km = st.slider("Distance to City (km)", 0.5, 80.0, 15.0)
        school_rating = st.slider("School Rating (1-10)", 1, 10, 6)
        crime_index = st.slider("Crime Index (0=safest, 100=highest)", 0, 100, 50)
        has_renovation = st.checkbox("Recently Renovated", value=False)

    house_age = 2024 - year_built

    input_dict = {
        "SqFtLiving": sqft_living,
        "Bedrooms": bedrooms,
        "Bathrooms": bathrooms,
        "LotSize": lot_size,
        "YearBuilt": year_built,
        "HouseAge": house_age,
        "OverallQuality": overall_quality,
        "GarageCars": garage_cars,
        "Stories": stories,
        "DistanceToCityKm": distance_to_city_km,
        "SchoolRating": school_rating,
        "CrimeIndex": crime_index,
        "HasRenovation": int(has_renovation),
    }
    input_df = pd.DataFrame([input_dict])
    input_df["SqFtPerBedroom"] = input_df["SqFtLiving"] / input_df["Bedrooms"]
    input_df["BathBedRatio"] = input_df["Bathrooms"] / input_df["Bedrooms"]
    input_df["LotToLivingRatio"] = input_df["LotSize"] / input_df["SqFtLiving"]
    input_df = input_df[feature_cols]

    prediction = model.predict(input_df)[0]

    st.divider()
    st.markdown(f"## Estimated Price: :green[${prediction:,.0f}]")

    rmse = metadata["results"][metadata["best_model"]]["RMSE"]
    st.caption(f"Typical model error (RMSE) is about ± ${rmse:,.0f}, based on holdout test performance.")

    similar = df[
        (df["Bedrooms"] == bedrooms)
        & (df["SqFtLiving"].between(sqft_living - 300, sqft_living + 300))
    ]
    if len(similar) > 0:
        st.caption(
            f"Comparable homes in the dataset ({len(similar)} found): "
            f"median price ${similar['MedHouseVal'].median():,.0f}"
        )
