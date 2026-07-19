from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier

app = FastAPI()

try:
    df = pd.read_csv('node_historical_telemetry.csv')
    X = df[['ph', 'tds', 'turbidity', 'temperature']]
    y = df['hazard_class']
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X, y)
    model_ready = True
except Exception as e:
    model_ready = False

class TelemetryData(BaseModel):
    ph: float
    tds: float
    turbidity: float
    temperature: float

@app.post("/analyze")
def analyze_water_telemetry(data: TelemetryData):
    if not model_ready:
        return {"risk_level": "Error", "info": "AI Model failed to initialize on the server."}
        
    vector = np.array([[data.ph, data.tds, data.turbidity, data.temperature]])
    predicted_class = clf.predict(vector)[0]
    
    diagnostic_matrix = {
        0: {"risk_level": "Optimal Stable", "info": "Catchment perimeter parameters are within potable baselines."},
        1: {"risk_level": "Moderate Alert", "info": "Elevated turbidity and TDS indicate localized soil erosion or mild surface runoff."},
        2: {"risk_level": "Critical High Risk", "info": "Acidic pH drop combined with thermal spikes flags an immediate chemical effluent breach."}
    }
    
    return diagnostic_matrix.get(predicted_class, {"risk_level": "Unknown", "info": "Data anomaly detected."})