

import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from pathlib import Path

# Project paths
BASE_DIR = Path(__file__).resolve().parent

# Page configuration
st.set_page_config(
    page_title="Customer Segmentation",
    page_icon="🛍️",
    layout="wide"
)

# Load saved files
@st.cache_resource
def load_models():
    model = joblib.load(BASE_DIR / "models" / "kmeans_model.pkl")
    scaler = joblib.load(BASE_DIR / "models" / "scaler.pkl")
    return model, scaler

@st.cache_data
def load_data():
    customers = pd.read_csv(BASE_DIR / "data" / "Customer_Segments.csv")
    summary = pd.read_csv(BASE_DIR / "data" / "Segment_Summary.csv")
    return customers, summary

model, scaler = load_models()
customers, summary = load_data()

# Customer segment descriptions
segment_descriptions = {
    0: "Older customers with relatively lower spending scores.",
    1: "Younger customers with relatively higher spending scores.",
    2: "Younger customers with relatively higher spending scores.",
    3: "Older customers with relatively lower spending scores."
}

# Website heading
st.title("🛍️ Customer Segmentation Dashboard")
st.write(
    "Discover customer groups using Machine Learning. "
    "Enter customer details to predict their segment."
)

st.divider()

# Dashboard metrics
col1, col2, col3 = st.columns(3)

col1.metric("Total Customers", len(customers))
col2.metric("Customer Segments", customers["Cluster"].nunique())
col3.metric("ML Algorithm", "K-Means")

st.divider()

# Prediction section
st.header("🔍 Predict Customer Segment")

with st.form("prediction_form"):
    col1, col2 = st.columns(2)

    with col1:
        age = st.number_input(
            "Customer Age", min_value=1, max_value=100, value=25
        )
        income = st.number_input(
            "Annual Income (k$)", min_value=1, max_value=200, value=60
        )

    with col2:
        spending = st.slider("Spending Score", 1, 100, 65)
        gender = st.selectbox("Gender", ["Female", "Male"])

    submitted = st.form_submit_button("Predict Customer Segment")

if submitted:
    gender_encoded = 1 if gender == "Male" else 0

    customer = pd.DataFrame([{
        "Age": age,
        "Annual Income (k$)": income,
        "Spending Score (1-100)": spending,
        "Gender_Encoded": gender_encoded
    }])

    customer_scaled = scaler.transform(customer)
    prediction = model.predict(customer_scaled)[0]

    st.success(
        f"Prediction completed! Customer belongs to Cluster {prediction}."
    )

    st.subheader("🎯 Customer Segment Details")

    st.write(segment_descriptions[prediction])

    st.info(
        "This prediction uses the trained K-Means model "
        "and the same preprocessing scaler."
    )

st.divider()

# Customer segment distribution
st.header("📊 Customer Segment Analysis")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Cluster Distribution")

    cluster_counts = customers["Cluster"].value_counts().sort_index()

    fig, ax = plt.subplots()
    ax.bar(cluster_counts.index.astype(str), cluster_counts.values)
    ax.set_xlabel("Cluster")
    ax.set_ylabel("Number of Customers")
    ax.set_title("Customers in Each Cluster")
    st.pyplot(fig)
    plt.close(fig)

with col2:
    st.subheader("Income vs Spending Score")

    fig, ax = plt.subplots()
    scatter = ax.scatter(
        customers["Annual Income (k$)"],
        customers["Spending Score (1-100)"],
        c=customers["Cluster"],
        cmap="viridis"
    )
    ax.set_xlabel("Annual Income (k$)")
    ax.set_ylabel("Spending Score")
    ax.set_title("Customer Segmentation")
    fig.colorbar(scatter, ax=ax, label="Cluster")
    st.pyplot(fig)
    plt.close(fig)

st.divider()

# Segment summary
st.header("📋 Customer Segment Summary")
st.dataframe(summary, use_container_width=True)

st.divider()

st.caption(
    "Customer Segmentation Using Machine Learning | SWYNEX Internship Project"
)
