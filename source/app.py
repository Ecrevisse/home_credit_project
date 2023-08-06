import pandas as pd
import mlflow
import uvicorn
from fastapi import FastAPI

# from .utils.model import CreditModelInput, CreditModel
from . import model as ml

# import utils.model as ml


app = FastAPI()
model = ml.CreditModel()


@app.get("/")
def index():
    return {"message": "Hello, stranger"}


@app.post("/predict")
def predict_species(input: ml.CreditModelInput):
    return model.predict(input)


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
