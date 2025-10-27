
import pickle
from fastapi import FastAPI
from pydantic import BaseModel

# Load the pipeline_v2.bin from the base image
with open('pipeline_v2.bin', 'rb') as f:
    pipeline = pickle.load(f)

app = FastAPI()

class Lead(BaseModel):
    lead_source: str
    number_of_courses_viewed: int
    annual_income: float

@app.get("/")
def home():
    return {"message": "Lead Scoring API (Docker)"}

@app.post("/predict")
def predict(lead: Lead):
    client = lead.dict()
    probability = pipeline.predict_proba([client])[0, 1]

    return {
        "probability": float(probability),
        "convert": bool(probability >= 0.5)
    }
