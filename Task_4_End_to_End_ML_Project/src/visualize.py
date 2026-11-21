
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# Project directory
BASE_DIR = Path(__file__).resolve().parent.parent

# Load customer segmentation results
df = pd.read_csv(BASE_DIR / "data" / "Customer_Segments.csv")

# Create scatter plot
plt.figure(figsize=(10, 6))

plt.scatter(
    df["Annual Income (k$)"],
    df["Spending Score (1-100)"],
    c=df["Cluster"],
    cmap="viridis",
    s=60
)

plt.title("Customer Segmentation Using K-Means")
plt.xlabel("Annual Income (k$)")
plt.ylabel("Spending Score (1-100)")

plt.colorbar(label="Cluster")
plt.grid(True, alpha=0.3)

# Save chart
output_path = BASE_DIR / "data" / "Customer_Segmentation_Plot.png"
plt.savefig(output_path, dpi=300, bbox_inches="tight")

print("Customer Segmentation Plot Saved!")
print("File Location:", output_path)

plt.show()


# Customer count in each cluster
cluster_counts = df["Cluster"].value_counts().sort_index()

plt.figure(figsize=(8, 5))

plt.bar(
    cluster_counts.index.astype(str),
    cluster_counts.values
)

plt.title("Number of Customers in Each Cluster")
plt.xlabel("Cluster")
plt.ylabel("Number of Customers")

plt.grid(axis="y", alpha=0.3)

# Save chart
count_plot_path = BASE_DIR / "data" / "Cluster_Distribution.png"
plt.savefig(count_plot_path, dpi=300, bbox_inches="tight")

print("\nCluster Distribution Chart Saved!")
print("File Location:", count_plot_path)

plt.show()
