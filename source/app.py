import pandas as pd
import mlflow
import uvicorn
from fastapi import FastAPI

# from model import CreditModelInput, CreditModel

app = FastAPI()
# model = CreditModel()


@app.get("/")
def index():
    return {"message": "Hello, stranger"}


# @app.post("/predict")
# def predict_species(input: CreditModelInput):
#     return model.predict(input)


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
