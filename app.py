from fastapi import FastAPI
import joblib
import numpy as np
from pydantic import BaseModel
import uvicorn

# 1. Initialize FastAPI app
app = FastAPI(title="Factory Sensor ML API", version="1.0")

# 2. Load the winning model saved from Notebook 2
model = joblib.load("model.pkl")

# 3. Define expected incoming data structure
class SensorInput(BaseModel):
    temperature_mean: float
    vibration_mean: float
    power_mean: float

@app.get("/")
def home():
    return {"message": "FastAPI ML Server is online and connected to model.pkl!"}

@app.post("/predict")
def predict_failure(data: SensorInput):
    features = np.array([[data.temperature_mean, data.vibration_mean, data.power_mean]])
    
    prediction = int(model.predict(features)[0])
    probability = float(model.predict_proba(features)[0][1]) if hasattr(model, "predict_proba") else None
    
    status = "Critical Failure (1)" if prediction == 1 else "Normal (0)"
    
    return {
        "prediction": prediction,
        "status": status,
        "failure_probability": probability
    }

if __name__ == "__main__":
    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=True)