import pandas as pd
from pathlib import Path

# Get the project directory
BASE_DIR = Path(__file__).resolve().parent.parent

# Load the dataset
data_path = BASE_DIR / "data" / "Mall_Customers.csv"

df = pd.read_csv(data_path)

# Display dataset information
print("Dataset loaded successfully!")

print("\nFirst 5 Records:")
print(df.head())

print("\nDataset Shape:")
print(df.shape)

print("\nColumn Names:")
print(df.columns.tolist())

print("\nDataset Information:")
print(df.info())

print("\nMissing Values:")
print(df.isnull().sum())

print("\nDataset Columns:")
print(df.columns.tolist())

print("\nDataset Shape:")
print(df.shape)

print("\nData Types:")
print(df.dtypes)

print("\nComplete Dataset Preview:")
print(df.head())

print("\nAll Dataset Columns:")
print(df.columns.tolist())

print("\nDataset Shape:")
print(df.shape)

# Select relevant features for customer segmentation

features = df[
    ["Age", "Annual Income (k$)", "Spending Score (1-100)"]
]

print("\nSelected Features:")
print(features.head())

print("\nFeature Dataset Shape:")
print(features.shape)

# Encode Gender

df["Gender_Encoded"] = df["Genre"].map({
    "Male": 1,
    "Female": 0
})

print("\nGender Encoding:")
print(df[["Genre", "Gender_Encoded"]].head())

print("\nGender Encoding Completed!")# Combine numerical features and encoded gender

X = df[
    [
        "Age",
        "Annual Income (k$)",
        "Spending Score (1-100)",
        "Gender_Encoded"
    ]
]

print("\nFinal Feature Dataset:")
print(X.head())

print("\nFinal Dataset Shape:", X.shape)

print("\nFinal Feature Columns:")
print(X.columns.tolist())

from sklearn.preprocessing import StandardScaler

# Initialize the scaler
scaler = StandardScaler()

# Scale the features
X_scaled = scaler.fit_transform(X)

# Convert scaled data into a DataFrame
X_scaled = pd.DataFrame(
    X_scaled,
    columns=X.columns
)

print("\nFeature Scaling Completed!")

print("\nScaled Dataset:")
print(X_scaled.head())

print("\nScaled Dataset Shape:", X_scaled.shape)

from sklearn.preprocessing import StandardScaler

# Initialize the scaler
scaler = StandardScaler()

# Scale the features
X_scaled = scaler.fit_transform(X)

# Convert scaled data into a DataFrame
X_scaled = pd.DataFrame(
    X_scaled,
    columns=X.columns
)

print("\nFeature Scaling Completed!")

print("\nScaled Dataset:")
print(X_scaled.head())

print("\nScaled Dataset Shape:", X_scaled.shape)

from sklearn.cluster import KMeans

# Initialize the K-Means model
kmeans = KMeans(
    n_clusters=4,
    random_state=42,
    n_init=10
)

# Train the model
kmeans.fit(X_scaled)

print("\nK-Means Model Training Completed!")

# Get cluster labels
cluster_labels = kmeans.labels_
# Add cluster labels to the original dataset
df["Cluster"] = cluster_labels

# Save customer segmentation results
output_path = BASE_DIR / "data" / "Customer_Segments.csv"
df.to_csv(output_path, index=False)

print("\nCustomer Segmentation Results Saved!")
print("File Location:", output_path)

print("\nCluster Assignments:")
print(cluster_labels)

print("\nNumber of Customers in Each Cluster:")
print(pd.Series(cluster_labels).value_counts().sort_index())

import joblib

# Save the trained model
model_path = BASE_DIR / "models" / "kmeans_model.pkl"

joblib.dump(kmeans, model_path)

print("\nK-Means Model Saved Successfully!")
print("Model Location:", model_path)

# Save the scaler
scaler_path = BASE_DIR / "models" / "scaler.pkl"

joblib.dump(scaler, scaler_path)

print("\nScaler Saved Successfully!")
print("Scaler Location:", scaler_path)

# Analyze customer segments
print("\nCustomer Segment Analysis:")

segment_summary = df.groupby("Cluster")[
    ["Age", "Annual Income (k$)", "Spending Score (1-100)"]
].mean().round(2)

print(segment_summary)

# Save customer segment summary
summary_path = BASE_DIR / "data" / "Segment_Summary.csv"
segment_summary.to_csv(summary_path)

print("\nSegment Summary Saved Successfully!")
print("File Location:", summary_path)