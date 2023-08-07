import pandas as pd
import mlflow
import uvicorn
from fastapi import FastAPI

import sys

sys.path.insert(0, "./source")

import model as ml

# import logging

# logger = logging.getLogger("azure")
# logger.setLevel(logging.DEBUG)

print("----> The logger works !!!!!!!   YEAH \o/")

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
