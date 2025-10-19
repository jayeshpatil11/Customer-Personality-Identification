# -*- coding: utf-8 -*-
"""
Created on Sun Oct 19 14:24:17 2025

@author: jayes
"""

# model.py — Backend logic

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score


# Load & Preprocess Dataset
def load_data(path="marketing_campaign.xlsx"):
    df = pd.read_excel(path)
    df['Income'] = df['Income'].fillna(df['Income'].median())

    df['Age'] = 2025 - df['Year_Birth']
    df['Total_Children'] = df['Kidhome'] + df['Teenhome']
    df['Total_Spent'] = (df['MntWines'] + df['MntFruits'] + df['MntMeatProducts'] +
                         df['MntFishProducts'] + df['MntSweetProducts'] + df['MntGoldProds'])
    df['Total_Accepted_Cmp'] = (df['AcceptedCmp1'] + df['AcceptedCmp2'] +
                                df['AcceptedCmp3'] + df['AcceptedCmp4'] + df['AcceptedCmp5'])
    df['Customer_For'] = (pd.to_datetime('2025-01-01') - pd.to_datetime(df['Dt_Customer'])).dt.days

    return df


# Prepare Features for Clustering
def prepare_features(df):
    features = ['Age', 'Income', 'Total_Spent', 'Recency', 'Total_Children']
    X = df[features]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled, features



# Apply KMeans Clustering
def perform_kmeans(df, X_scaled, k=4):
    kmeans = KMeans(n_clusters=k, random_state=42)
    df['Cluster'] = kmeans.fit_predict(X_scaled)
    sil_score = silhouette_score(X_scaled, df['Cluster'])
    return df, sil_score



# Apply Agglomerative Clustering
def perform_agglomerative(df, X_scaled, n_clusters=4):
    agg = AgglomerativeClustering(n_clusters=n_clusters, linkage='ward')
    df['Agg_Cluster'] = agg.fit_predict(X_scaled)
    return df
