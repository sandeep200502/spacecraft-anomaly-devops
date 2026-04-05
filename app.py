from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
import numpy as np
import joblib

# ================================
# 1. Initialize App
# ================================
app = FastAPI(title="🚀 Spacecraft Anomaly Detection API")

# ================================
# 2. Load Model & Scaler
# ================================
try:
    model = joblib.load("model.pkl")
    scaler = joblib.load("scaler.pkl")
    print("✅ Model and scaler loaded successfully")
except Exception as e:
    print("❌ Error loading model:", str(e))
    model = None
    scaler = None

# ================================
# 3. Define Input Schema
# ================================
class InputData(BaseModel):
    data: List[float]

# ================================
# 4. Root Endpoint
# ================================
@app.get("/")
def home():
    return {"message": "🚀 Spacecraft Anomaly Detection API Running"}

# ================================
# 5. Prediction Endpoint
# ================================
@app.post("/predict")
def predict(input_data: InputData):
    try:
        # Check model loaded
        if model is None or scaler is None:
            return {"error": "Model not loaded properly"}

        print("📥 Received input:", input_data.data)

        # Validate input length
        if len(input_data.data) != 2:
            return {"error": "Input must contain exactly 2 values (channel_1, channel_2)"}

        # Convert to numpy
        data = np.array(input_data.data).reshape(1, -1)
        print("🔹 Before scaling:", data)

        # Scale
        data_scaled = scaler.transform(data)
        print("🔹 After scaling:", data_scaled)

        # Predict
        pred = model.predict(data_scaled)
        print("🔹 Raw prediction:", pred)

        # Convert to anomaly label
        anomaly = int(pred[0] == -1)

        return {
            "input": input_data.data,
            "anomaly": anomaly,
            "status": "success"
        }

    except Exception as e:
        print("❌ ERROR:", str(e))
        return {
            "error": str(e),
            "status": "failed"
        }
