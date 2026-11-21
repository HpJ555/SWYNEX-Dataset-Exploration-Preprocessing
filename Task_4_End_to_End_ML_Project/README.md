# Customer Segmentation Using Machine Learning

## Project Overview

This project uses Machine Learning to segment mall customers into different groups based on their age, annual income, spending score, and gender.

The main objective is to identify customer groups with similar characteristics using the K-Means Clustering algorithm.

## Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Joblib

## Dataset Description

The project uses the Mall Customers dataset containing 200 customer records.

**Features used:**
- Age
- Annual Income (k$)
- Spending Score (1-100)
- Gender (encoded as numerical values)

## Project Workflow

1. Data Loading
2. Data Exploration
3. Data Preprocessing
4. Gender Encoding
5. Feature Scaling using StandardScaler
6. K-Means Model Training
7. Model Evaluation
8. Customer Segmentation
9. New Customer Prediction
10. Data Visualization

## Machine Learning Algorithm

**K-Means Clustering**

K-Means is an unsupervised machine learning algorithm that groups similar data points into clusters.

This project uses K-Means with 4 clusters and a random state of 42.

## Model Evaluation

The model was evaluated using three metrics:

| Metric | Result |
|---|---:|
| Silhouette Score | 0.2984 |
| Davies-Bouldin Score | 1.2807 |
| Calinski-Harabasz Score | 69.1258 |

## Cluster Distribution

| Cluster | Number of Customers |
|---|---:|
| Cluster 0 | 56 |
| Cluster 1 | 46 |
| Cluster 2 | 56 |
| Cluster 3 | 42 |
| **Total** | **200** |

## Project Structure

```text
Task_4_End_to_End_ML_Project/
│
├── data/
│   ├── Mall_Customers.csv
│   ├── Customer_Segments.csv
│   ├── Segment_Summary.csv
│   ├── Customer_Segmentation_Plot.png
│   └── Cluster_Distribution.png
│
├── models/
│   ├── kmeans_model.pkl
│   └── scaler.pkl
│
├── src/
│   ├── train_model.py
│   ├── predict.py
│   └── visualize.py
│
├── requirements.txt
└── README.md
```

## Installation

Install the required Python libraries:

```bash
pip install -r requirements.txt
```

## How to Run the Project

**1. Train the Machine Learning Model**

```bash
python src/train_model.py
```

**2. Predict a New Customer Segment**

```bash
python src/predict.py
```

**3. Generate Visualizations**

```bash
python src/visualize.py
```

Run these commands from inside the project folder.

## Sample Prediction

Input:

- Age: 25
- Annual Income: 60 k$
- Spending Score: 65
- Gender: Female

Output:

```text
Prediction Completed!
Customer belongs to Cluster: 2
```

## Visualizations

The project generates two visualizations:

1. Customer Segmentation Scatter Plot
2. Cluster Distribution Bar Chart

These charts help understand the distribution of customers across the four clusters.

## Conclusion

This project demonstrates an end-to-end machine learning workflow, including data preprocessing, model training, evaluation, customer segmentation, prediction, and visualization.

The trained model can predict the cluster of a new customer based on the provided customer details.

## Future Enhancements

- Develop an interactive Streamlit dashboard.
- Experiment with additional clustering algorithms.
- Improve customer segment interpretation.
- Add interactive data visualizations.

---

**Project:** Customer Segmentation Using Machine Learning  
**Internship:** SWYNEX Technologies