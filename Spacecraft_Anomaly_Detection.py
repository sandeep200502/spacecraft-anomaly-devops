#!/usr/bin/env python
# coding: utf-8

import numpy as np
import pandas as pd
import joblib

from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

# ================================
# 1. GENERATE SAMPLE TELEMETRY DATA
# ================================

np.random.seed(42)

n_samples = 1000
time_idx = np.arange(n_samples)

# Simulated spacecraft telemetry
channel_1 = np.sin(time_idx * 0.01) + np.random.normal(0, 0.1, n_samples)
channel_2 = np.cos(time_idx * 0.005) + np.random.normal(0, 0.15, n_samples)

# Inject anomaly
anomaly_start, anomaly_end = 400, 500
channel_1[anomaly_start:anomaly_end] += 3.0
channel_2[anomaly_start:anomaly_end] *= 0.1

telemetry = pd.DataFrame({
    "channel_1": channel_1,
    "channel_2": channel_2
})

# True labels
y_true = np.zeros(n_samples)
y_true[anomaly_start:anomaly_end] = 1

print("✅ Data prepared")

# ================================
# 2. PREPROCESSING
# ================================

X = telemetry.values

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print("✅ Data scaled")

# ================================
# 3. TRAIN MODEL (Isolation Forest)
# ================================

model = IsolationForest(contamination=0.05, random_state=42)
model.fit(X_scaled)

print("✅ Model trained")

# ================================
# 4. SAVE MODEL (IMPORTANT)
# ================================

joblib.dump(model, "model.pkl")
joblib.dump(scaler, "scaler.pkl")

print("🚀 Model saved as model.pkl and scaler.pkl")
