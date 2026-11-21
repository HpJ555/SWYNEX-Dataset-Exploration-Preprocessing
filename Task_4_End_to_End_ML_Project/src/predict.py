
import pandas as pd
import joblib
from pathlib import Path

# Project directory
BASE_DIR = Path(__file__).resolve().parent.parent

# Load saved model and scaler
model = joblib.load(BASE_DIR / "models" / "kmeans_model.pkl")
scaler = joblib.load(BASE_DIR / "models" / "scaler.pkl")

print("Customer Segmentation Prediction")
print("--------------------------------")

# Get customer details
age = int(input("Enter customer age: "))
income = float(input("Enter annual income (k$): "))
score = float(input("Enter spending score (1-100): "))
gender = input("Enter gender (Male/Female): ").strip().capitalize()

if gender not in ["Male", "Female"]:
    raise ValueError("Gender must be Male or Female.")

# Prepare customer data
customer = pd.DataFrame(
    [[age, income, score, 1 if gender == "Male" else 0]],
    columns=[
        "Age",
        "Annual Income (k$)",
        "Spending Score (1-100)",
        "Gender_Encoded"
    ]
)

# Apply the saved scaler
customer_scaled = pd.DataFrame(
    scaler.transform(customer),
    columns=customer.columns
)
# Predict customer segment
prediction = model.predict(customer_scaled)

print("\nPrediction Completed!")
print("Customer belongs to Cluster:", prediction[0])
