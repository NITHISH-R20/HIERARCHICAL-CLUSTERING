import streamlit as st
import numpy as np
from sklearn.datasets import load_wine
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import AgglomerativeClustering

# ----------------------------------
# Page Configuration
# ----------------------------------
st.set_page_config(page_title="Hierarchical Clustering")

st.title("Hierarchical Clustering Application")
st.write("Simple Input → Output Interface")

# ----------------------------------
# Load Dataset (Hidden)
# ----------------------------------
wine = load_wine()
X = wine.data

# Standardization
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ----------------------------------
# User Inputs
# ----------------------------------
st.subheader("Input Parameters")

n_clusters = st.number_input(
    "Enter number of clusters",
    min_value=2,
    max_value=10,
    value=3
)

linkage = st.selectbox(
    "Select linkage method",
    ["ward", "complete", "average", "single"]
)

# ----------------------------------
# Run Clustering
# ----------------------------------
if st.button("Run Hierarchical Clustering"):
    model = AgglomerativeClustering(
        n_clusters=n_clusters,
        linkage=linkage
    )

    labels = model.fit_predict(X_scaled)

    unique_clusters = np.unique(labels)

    # ----------------------------------
    # Output
    # ----------------------------------
    st.subheader("Output Results")

    st.write("Total Data Points :", len(labels))
    st.write("Number of Clusters :", len(unique_clusters))

    for cluster in unique_clusters:
        count = np.sum(labels == cluster)
        st.write(f"Cluster {cluster} count :", count)
