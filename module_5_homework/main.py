

from fastapi import FastAPI
from pydantic import BaseModel
import pickle

with open("pipeline_v1.bin", "rb") as f_in:
    dv, model = pickle.load(f_in)

app = FastAPI()

class Record(BaseModel):
    lead_source: str
    number_of_courses_viewed: int
    annual_income: float

@app.post("/predict")
def predict(record: Record):
    X = dv.transform([record.dict()])
    y_pred = model.predict_proba(X)[0, 1]

    return {"subscription_probability": float(y_pred)}
