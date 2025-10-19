# -*- coding: utf-8 -*-
"""
Created on Sun Oct 19 14:23:25 2025

@author: jayes
"""


# app.py — Streamlit Dashboard

import streamlit as st
import plotly.express as px
import pandas as pd
from model import load_data, prepare_features, perform_kmeans, perform_agglomerative

# Streamlit settings
st.set_page_config(page_title="Customer Personality Dashboard", layout="wide")

# Title
st.title("🧩 Customer Personality Analysis Dashboard")
st.markdown("Explore EDA, K-Means & Agglomerative Clustering interactively.")


# Sidebar
st.sidebar.header("⚙️ Controls")

uploaded_file = st.sidebar.file_uploader("Upload your dataset (.xlsx)", type=["xlsx"])
if uploaded_file:
    df = load_data(uploaded_file)
else:
    df = load_data("marketing_campaign.xlsx")
    st.sidebar.info("Using default dataset: marketing_campaign.xlsx")

# Filters
min_income, max_income = int(df['Income'].min()), int(df['Income'].max())
income_range = st.sidebar.slider("Select Income Range", min_income, max_income, (min_income, max_income))
df = df[(df['Income'] >= income_range[0]) & (df['Income'] <= income_range[1])]


# Metrics Cards
st.markdown("### 📊 Key Metrics")

col1, col2, col3, col4 = st.columns(4)
col1.metric("👥 Total Customers", len(df))
col2.metric("💰 Avg Income", f"${df['Income'].mean():.0f}")
col3.metric("🛍️ Avg Total Spending", f"${df['Total_Spent'].mean():.0f}")
col4.metric("📅 Avg Recency", f"{df['Recency'].mean():.0f} days")


# Tabs
tab1, tab2, tab3 = st.tabs(["📈 EDA", "🤖 K-Means Clustering", "🌳 Agglomerative Clustering"])


# TAB 1: EDA
with tab1:
    st.subheader("📊 Exploratory Data Analysis")

    fig1 = px.histogram(df, x="Age", nbins=30, title="Age Distribution", color_discrete_sequence=['teal'])
    st.plotly_chart(fig1, use_container_width=True)

    fig2 = px.histogram(df, x="Income", nbins=30, title="Income Distribution", color_discrete_sequence=['coral'])
    st.plotly_chart(fig2, use_container_width=True)

    fig3 = px.scatter(df, x="Income", y="Total_Spent", color="Education",
                      title="Income vs Spending by Education", color_discrete_sequence=px.colors.qualitative.Set2)
    st.plotly_chart(fig3, use_container_width=True)


# TAB 2: K-MEANS CLUSTERING
with tab2:
    st.subheader("🤖 K-Means Clustering")

    X_scaled, features = prepare_features(df)
    k = st.slider("Select number of clusters (k):", 2, 10, 4)
    df, sil_score = perform_kmeans(df, X_scaled, k)

    st.success(f"✅ Silhouette Score for k={k}: {sil_score:.3f}")

    fig4 = px.scatter(df, x="Income", y="Total_Spent", color="Cluster",
                      color_continuous_scale="Viridis", title="K-Means Clustering: Income vs Spending")
    st.plotly_chart(fig4, use_container_width=True)

    st.write("### 📋 Cluster Profile")
    st.dataframe(df.groupby('Cluster')[features].mean().style.highlight_max(axis=0, color='lightgreen'))

    # Download button
    st.download_button(
        label="💾 Download K-Means Clustered Data",
        data=df.to_csv(index=False).encode('utf-8'),
        file_name='kmeans_clusters.csv',
        mime='text/csv'
    )

# TAB 3: AGGLOMERATIVE CLUSTERING

with tab3:
    st.subheader("🌳 Agglomerative Clustering")

    n_clusters = st.slider("Select number of clusters:", 2, 10, 4, key="agg")
    df = perform_agglomerative(df, X_scaled, n_clusters)

    fig5 = px.scatter(df, x="Income", y="Total_Spent", color="Agg_Cluster",
                      color_continuous_scale="Plasma", title="Agglomerative Clustering: Income vs Spending")
    st.plotly_chart(fig5, use_container_width=True)

    st.write("### 📋 Cluster Summary")
    st.dataframe(df.groupby('Agg_Cluster')[features].mean().style.highlight_max(axis=0, color='lightblue'))

    st.download_button(
        label="💾 Download Agglomerative Cluster Data",
        data=df.to_csv(index=False).encode('utf-8'),
        file_name='agglomerative_clusters.csv',
        mime='text/csv'
    )
