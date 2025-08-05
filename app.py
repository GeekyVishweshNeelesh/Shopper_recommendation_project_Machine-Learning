import streamlit as st
import pandas as pd
import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load CSV files
rfm_df = pd.read_csv("/home/vishwesh/Documents/Labmentix_Internship/Project_4/rfm_scaled.csv")
product_df = pd.read_csv("/home/vishwesh/Documents/Labmentix_Internship/Project_4/product_data.csv")

# Check for required columns
required_rfm_cols = {'Recency_scaled', 'Frequency_scaled', 'Monetary_scaled'}
if not required_rfm_cols.issubset(rfm_df.columns):
    st.error("rfm_scaled.csv must contain Recency_scaled, Frequency_scaled, and Monetary_scaled columns")
    st.stop()

if not {'CustomerID', 'ProductName'}.issubset(product_df.columns):
    st.error("product_data.csv must contain CustomerID and ProductName columns")
    st.stop()

# Remove duplicates and reset
product_df.dropna(subset=['ProductName'], inplace=True)
product_df.drop_duplicates(subset='ProductName', inplace=True)
product_df.reset_index(drop=True, inplace=True)

# Segment label map
segment_map = {
    0: 'High-Value',
    1: 'Regular',
    2: 'Occasional',
    3: 'At-Risk'
}

# Sidebar Navigation
st.sidebar.title("🔍 Navigation")
page = st.sidebar.radio("Go to", ["Product Recommendation", "Customer Segmentation"])

# ---------------------- PRODUCT RECOMMENDATION ----------------------
if page == "Product Recommendation":
    st.title("🎯 Product Recommendation Engine")

    product_name = st.text_input("📥 Enter Product Name")

    if st.button("🔎 Get Recommendations"):
        if product_name.strip() == "":
            st.warning("Please enter a product name.")
        else:
            tfidf = TfidfVectorizer(stop_words='english')
            tfidf_matrix = tfidf.fit_transform(product_df['ProductName'])

            if product_name not in product_df['ProductName'].values:
                st.error("❌ Product not found. Please enter a valid product name.")
            else:
                idx = product_df[product_df['ProductName'] == product_name].index[0]
                cosine_sim = cosine_similarity(tfidf_matrix[idx], tfidf_matrix).flatten()
                similar_indices = cosine_sim.argsort()[-6:][::-1]

                recommended_products = product_df.iloc[similar_indices[1:]]['ProductName'].values

                st.markdown("### ✅ Recommended Products:")
                for i, prod in enumerate(recommended_products, start=1):
                    st.markdown(
                        f"""
                        <div style="background-color:black;padding:15px;margin-bottom:10px;border-radius:10px">
                            <h4 style='color:white;'>📌 {i}. {prod}</h4>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

# ---------------------- CUSTOMER SEGMENTATION ----------------------
elif page == "Customer Segmentation":
    st.title("🎯 Customer Segmentation")

    st.markdown("Enter the following RFM values:")

    recency = st.number_input("📅 Recency (in days)", min_value=0)
    frequency = st.number_input("🔁 Frequency (number of purchases)", min_value=0)
    monetary = st.number_input("💰 Monetary (total spend)", min_value=0.0)

    model_choice = st.selectbox("Select Model", ["KMeans", "Agglomerative", "DBSCAN"])

    if st.button("🔍 Predict Cluster"):
        features = [[recency, frequency, monetary]]

        try:
            if model_choice == "KMeans":
                model = joblib.load("/home/vishwesh/Documents/Labmentix_Internship/Project_4/kmeans_model.joblib")
                label = model.predict(features)[0]
                segment_label = segment_map.get(label, f"Segment {label}")
                st.markdown(f"<h3 style='color: green;'>📌 This customer belongs to: <b>{segment_label}</b></h3>", unsafe_allow_html=True)

            elif model_choice == "Agglomerative":
                st.warning("⚠️ Agglomerative Clustering does not support prediction for new data points. Please use KMeans.")

            elif model_choice == "DBSCAN":
                st.warning("⚠️ DBSCAN does not support prediction for new data points. Please use KMeans.")

        except Exception as e:
            st.error(f"Error: {str(e)}")
