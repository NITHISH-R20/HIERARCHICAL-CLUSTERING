import streamlit as st
import pandas as pd
from sklearn.datasets import load_wine
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN

# -----------------------------
# Page Configuration
# -----------------------------
st.set_page_config(page_title="Clustering App")

st.title("🍷 Wine Clustering App")
st.write("Simple Input → Output Interface")

# -----------------------------
# Load & Prepare Dataset (Hidden)
# -----------------------------
wine = load_wine()
X = wine.data

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# -----------------------------
# User Inputs
# -----------------------------
st.subheader("🔧 Enter DBSCAN Parameters")

eps = st.number_input("Enter eps value", min_value=0.1, max_value=10.0, value=2.0, step=0.1)
min_samples = st.number_input("Enter min_samples", min_value=1, max_value=20, value=2)

# -----------------------------
# Button Action
# -----------------------------
if st.button("Run Clustering"):
    model = DBSCAN(eps=eps, min_samples=min_samples)
    labels = model.fit_predict(X_scaled)

    unique_clusters = set(labels)
    noise_count = list(labels).count(-1)

    st.subheader("📌 Output")

    st.write("Total Clusters Found:", len(unique_clusters) - (1 if -1 in unique_clusters else 0))
    st.write("Noise Points:", noise_count)
    st.write("Total Data Points:", len(labels))
