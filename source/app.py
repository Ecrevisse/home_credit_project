import pandas as pd
import mlflow
import uvicorn
from fastapi import FastAPI

import sys

sys.path.insert(0, "./source")

import model as ml


app = FastAPI()
model = ml.CreditModel()


@app.get("/")
def index():
    return {"message": "Hello, stranger"}


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
def predict_species(input: ml.CreditModelInput):
    return model.predict(input)


@app.post("/shap_local")
def shapSummary(input: ml.CreditModelInput):
    return model.shap_local(input)


@app.get("/shap_global")
def shapSummary():
    return model.shap_global()


@app.post("/client_info")
def client_info(input: ml.ClientInfoInput):
    return model.client_info(input)


@app.get("/model_features")
def model_features():
    return model.model_features()


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
