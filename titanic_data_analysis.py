# Titanic Data Analysis Dashboard
# Google Colab / VS Code compatible
#
# Run:
#   pip install streamlit pandas plotly
#   streamlit run titanic_dashboard.py
#
# The dashboard asks you to IMPORT the Titanic CSV dataset.
# It does not contain a hard-coded Titanic dataset.

import streamlit as st
import pandas as pd
import plotly.express as px

st.set_page_config(
    page_title="Titanic Data Analysis Dashboard",
    page_icon="🚢",
    layout="wide"
)

st.title("🚢 Titanic Data Analysis Dashboard")
st.caption("Import your Titanic CSV dataset to perform interactive data analysis.")

# ---------------------------------------------------------
# DATA IMPORT
# ---------------------------------------------------------

st.sidebar.header("📂 Import Titanic Dataset")

uploaded_file = st.sidebar.file_uploader(
    "Upload Titanic CSV file",
    type=["csv"]
)

if uploaded_file is None:
    st.info(
        "Please upload your Titanic CSV dataset from the sidebar to start the dashboard."
    )
    st.markdown("""
    ### Expected Titanic columns

    Your dataset should normally contain:

    - `PassengerId`
    - `Survived`
    - `Pclass`
    - `Name`
    - `Sex`
    - `Age`
    - `SibSp`
    - `Parch`
    - `Fare`
    - `Cabin`
    - `Embarked`

    The standard Kaggle Titanic `train.csv` works with this dashboard.
    """)
    st.stop()

try:
    df = pd.read_csv(uploaded_file)
except Exception as e:
    st.error("Unable to read the uploaded CSV file.")
    st.exception(e)
    st.stop()

# ---------------------------------------------------------
# COLUMN VALIDATION
# ---------------------------------------------------------

required_columns = [
    "Survived",
    "Pclass",
    "Sex",
    "Age",
    "SibSp",
    "Parch",
    "Fare"
]

missing_columns = [col for col in required_columns if col not in df.columns]

if missing_columns:
    st.error("The uploaded dataset is missing required columns:")
    st.write(missing_columns)
    st.stop()

# ---------------------------------------------------------
# SIDEBAR FILTERS
# ---------------------------------------------------------

st.sidebar.header("🔎 Filters")

sex_values = sorted(df["Sex"].dropna().astype(str).unique())

selected_sex = st.sidebar.multiselect(
    "Gender",
    sex_values,
    default=sex_values
)

class_values = sorted(df["Pclass"].dropna().unique())

selected_classes = st.sidebar.multiselect(
    "Passenger Class",
    class_values,
    default=class_values
)

if "Embarked" in df.columns:
    embarked_values = sorted(df["Embarked"].dropna().astype(str).unique())

    selected_embarked = st.sidebar.multiselect(
        "Embarked",
        embarked_values,
        default=embarked_values
    )
else:
    selected_embarked = []

age_min = float(df["Age"].min())
age_max = float(df["Age"].max())

age_range = st.sidebar.slider(
    "Age Range",
    min_value=float(int(age_min)),
    max_value=float(int(age_max)),
    value=(float(int(age_min)), float(int(age_max)))
)

# ---------------------------------------------------------
# FILTER DATA
# ---------------------------------------------------------

filtered_df = df[
    df["Sex"].astype(str).isin(selected_sex)
    & df["Pclass"].isin(selected_classes)
    & df["Age"].fillna(df["Age"].median()).between(
        age_range[0],
        age_range[1]
    )
].copy()

if "Embarked" in df.columns:
    filtered_df = filtered_df[
        filtered_df["Embarked"].astype(str).isin(selected_embarked)
    ]

# ---------------------------------------------------------
# KPI CALCULATIONS
# ---------------------------------------------------------

total_passengers = len(filtered_df)

survivors = int(filtered_df["Survived"].sum())

deaths = total_passengers - survivors

survival_rate = (
    survivors / total_passengers * 100
    if total_passengers > 0
    else 0
)

average_age = (
    filtered_df["Age"].mean()
    if total_passengers > 0
    else 0
)

average_fare = (
    filtered_df["Fare"].mean()
    if total_passengers > 0
    else 0
)

# ---------------------------------------------------------
# KPI DASHBOARD
# ---------------------------------------------------------

st.subheader("📊 Key Performance Indicators")

col1, col2, col3, col4, col5 = st.columns(5)

col1.metric(
    "👥 Passengers",
    f"{total_passengers:,}"
)

col2.metric(
    "🛟 Survivors",
    f"{survivors:,}"
)

col3.metric(
    "💀 Deaths",
    f"{deaths:,}"
)

col4.metric(
    "📈 Survival Rate",
    f"{survival_rate:.1f}%"
)

col5.metric(
    "🎂 Average Age",
    f"{average_age:.1f}"
)

st.divider()

# ---------------------------------------------------------
# SURVIVAL DISTRIBUTION
# ---------------------------------------------------------

col1, col2 = st.columns(2)

with col1:

    st.subheader("🛟 Survival Distribution")

    survival_data = (
        filtered_df["Survived"]
        .map({
            0: "Did Not Survive",
            1: "Survived"
        })
        .value_counts()
        .reset_index()
    )

    survival_data.columns = ["Outcome", "Passengers"]

    fig = px.pie(
        survival_data,
        names="Outcome",
        values="Passengers",
        hole=0.45,
        title="Survival vs Death"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

# ---------------------------------------------------------
# SURVIVAL BY GENDER
# ---------------------------------------------------------

with col2:

    st.subheader("👩👨 Survival by Gender")

    gender_data = (
        filtered_df
        .groupby("Sex", as_index=False)["Survived"]
        .mean()
    )

    gender_data["Survival Rate (%)"] = (
        gender_data["Survived"] * 100
    )

    fig = px.bar(
        gender_data,
        x="Sex",
        y="Survival Rate (%)",
        color="Sex",
        text="Survival Rate (%)",
        title="Survival Rate by Gender"
    )

    fig.update_traces(
        texttemplate="%{text:.1f}%",
        textposition="outside"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

# ---------------------------------------------------------
# SURVIVAL BY CLASS
# ---------------------------------------------------------

col1, col2 = st.columns(2)

with col1:

    st.subheader("🎟️ Survival by Passenger Class")

    class_data = (
        filtered_df
        .groupby("Pclass", as_index=False)["Survived"]
        .mean()
    )

    class_data["Survival Rate (%)"] = (
        class_data["Survived"] * 100
    )

    fig = px.bar(
        class_data,
        x="Pclass",
        y="Survival Rate (%)",
        color="Pclass",
        text="Survival Rate (%)",
        title="Survival Rate by Passenger Class"
    )

    fig.update_traces(
        texttemplate="%{text:.1f}%",
        textposition="outside"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

# ---------------------------------------------------------
# AGE DISTRIBUTION
# ---------------------------------------------------------

with col2:

    st.subheader("🎂 Age Distribution")

    fig = px.histogram(
        filtered_df,
        x="Age",
        color="Survived",
        nbins=30,
        barmode="overlay",
        title="Age Distribution by Survival"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

# ---------------------------------------------------------
# FARE ANALYSIS
# ---------------------------------------------------------

st.subheader("💰 Fare Distribution by Passenger Class")

fig = px.box(
    filtered_df,
    x="Pclass",
    y="Fare",
    color="Pclass",
    points="outliers",
    title="Passenger Fare Distribution"
)

st.plotly_chart(
    fig,
    use_container_width=True
)

# ---------------------------------------------------------
# FAMILY SIZE ANALYSIS
# ---------------------------------------------------------

st.subheader("👨‍👩‍👧 Family Size vs Survival")

family_df = filtered_df.copy()

family_df["FamilySize"] = (
    family_df["SibSp"].fillna(0)
    + family_df["Parch"].fillna(0)
    + 1
)

family_data = (
    family_df
    .groupby("FamilySize", as_index=False)["Survived"]
    .mean()
)

family_data["Survival Rate (%)"] = (
    family_data["Survived"] * 100
)

# Keep chart readable
family_data = family_data[
    family_data["FamilySize"] <= 10
]

fig = px.line(
    family_data,
    x="FamilySize",
    y="Survival Rate (%)",
    markers=True,
    title="Survival Rate by Family Size"
)

st.plotly_chart(
    fig,
    use_container_width=True
)

# ---------------------------------------------------------
# CORRELATION HEATMAP
# ---------------------------------------------------------

st.subheader("🔥 Correlation Analysis")

numeric_columns = filtered_df.select_dtypes(
    include=["number"]
).columns

if len(numeric_columns) >= 2:

    correlation = filtered_df[numeric_columns].corr()

    fig = px.imshow(
        correlation,
        text_auto=".2f",
        aspect="auto",
        title="Correlation Matrix"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

# ---------------------------------------------------------
# DATASET INFORMATION
# ---------------------------------------------------------

st.subheader("📋 Dataset Information")

info_col1, info_col2 = st.columns(2)

with info_col1:

    st.write("**Rows:**", df.shape[0])
    st.write("**Columns:**", df.shape[1])

with info_col2:

    missing_values = df.isnull().sum().sum()

    st.write(
        "**Total Missing Values:**",
        int(missing_values)
    )

# ---------------------------------------------------------
# FILTERED DATA
# ---------------------------------------------------------

st.subheader("📄 Filtered Passenger Data")

st.dataframe(
    filtered_df,
    use_container_width=True,
    hide_index=True
)

# ---------------------------------------------------------
# DOWNLOAD
# ---------------------------------------------------------

csv_data = filtered_df.to_csv(
    index=False
).encode("utf-8")

st.download_button(
    label="⬇️ Download Filtered Dataset",
    data=csv_data,
    file_name="filtered_titanic_data.csv",
    mime="text/csv"
)

st.divider()

st.caption(
    "Titanic Data Analysis Dashboard | "
    "Python • Pandas • Plotly • Streamlit"
)
