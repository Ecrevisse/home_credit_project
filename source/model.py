import numpy as np
import pandas as pd
import mlflow
from mlflow import MlflowClient
from pydantic import BaseModel
import gc
import os
import shap
import psutil
import sys
from pympler import asizeof


def show_memory(prefix=""):
    process = psutil.Process(os.getpid())
    print(f"{prefix} -> RAM Used (MB): {process.memory_info().rss / 1024 ** 2}")

    # objs = gc.get_objects()
    # print(asizeof.asizeof(objs))
    # s = sys.getsizeof(objs)
    # sizes = []
    # for i, o in enumerate(objs):
    #     sizes.append(asizeof.asizeof(o))
    #     s += asizeof.asizeof(o)
    # sizes.sort()

    # print(f"{prefix} -> Number of objects: {len(objs)}")
    # print(f"{prefix} -> Size of objects: {s / 1024 ** 2:.2f} MB")


class CreditModelInput(BaseModel):
    client_id: int


class ClientInfoInput(BaseModel):
    client_id: int
    client_infos: list[str]


class CreditModel:
    def __init__(self):
        run_id = "0cb98077b68b4bba8ea208b79cb1d2e8"
        logged_model = f"runs:/{run_id}/model"

        client = MlflowClient()
        run = client.get_run(run_id)

        if run.info.status != "FINISHED":
            raise Exception(f"Model with status {run.info.status}")
        if "count_cols" not in run.data.params:
            raise Exception(f"Wrong model, doesn't have count_cols")
        if "threshold" not in run.data.params:
            raise Exception(f"Wrong model, doesn't have threshold")

        python_model_for_signature = mlflow.pyfunc.load_model(logged_model)
        self.model = mlflow.lightgbm.load_model(logged_model)
        cols = []
        for elem in python_model_for_signature._model_meta._signature.inputs.to_dict():
            cols.append(elem["name"])
        print(f"model_signature: {sys.getsizeof(python_model_for_signature)}")
        print(f"model: {sys.getsizeof(self.model)}")
        self.count_cols = int(run.data.params["count_cols"])
        self.threshold = float(run.data.params["threshold"])

        print("loading data...")
        self.data = self._load_and_prepare_data(cols)
        print(self.data.shape)
        print("model ready")

        self._load_and_prep_explainer()

    def _load_and_prepare_data(self, cols):
        df = pd.read_feather("./input/test_cleaned.feather")

        test_df = df[df["TARGET"].isnull()]
        test_df = test_df.fillna(0)
        test_df = test_df.replace([np.inf, -np.inf], 0)

        cols.append("SK_ID_CURR")
        test_df = test_df[cols]
        return test_df

    def _load_and_prep_explainer(self):
        df = pd.read_feather("./input/valid_cleaned.feather")

        df = df[:1000]
        df = df.fillna(0)
        df = df.replace([np.inf, -np.inf], 0)

        for col in df.columns:
            if df[col].dtype == "bool":
                df[col] = df[col].astype(int)

        df = df[self.data.columns]
        self.valid_data = df
        self.explainer = shap.TreeExplainer(self.model)
        del df

    def predict(self, input: CreditModelInput):
        X = self.data.loc[self.data["SK_ID_CURR"] == input.client_id]

        if len(X) == 0:
            raise Exception(f"Client ID {input.client_id} doesn't exist")
        X = X.drop(["SK_ID_CURR"], axis=1)
        pred = self.model.predict_proba(
            X,
            num_iteration=self.model.best_iteration_,
        )[:, 1]
        del X
        gc.collect()
        if len(pred) != 1:
            raise Exception(f"An issue occurred with the prediction")
        return 1 if (pred[0] > self.threshold) else 0, pred[0], self.threshold

    def shap_local(self, input: CreditModelInput):
        X = self.data.loc[self.data["SK_ID_CURR"] == input.client_id]
        if len(X) == 0:
            raise Exception(f"Client ID {input.client_id} doesn't exist")
        X = X.drop(["SK_ID_CURR"], axis=1)
        shap_values = self.explainer(X)[0][:, 1]
        return {
            "shap_values": shap_values.values.tolist(),
            "base_value": shap_values.base_values,
            "data": X.values.tolist(),
            "feature_names": X.columns.tolist(),
        }

    def shap_global(self):
        X = self.valid_data
        X = X.drop(["SK_ID_CURR"], axis=1)
        for col in X.columns:
            if X[col].dtype == "bool":
                X[col] = X[col].astype(int)

        shap_values = self.explainer(
            X,
        )[:, :, 1]
        return {
            "shap_values": shap_values.values.tolist(),
            "base_value": shap_values.base_values.tolist(),
            "data": shap_values.data.tolist(),
            "feature_names": X.columns.tolist(),
        }

    def client_info(self, input: ClientInfoInput):
        X = self.data.loc[self.data["SK_ID_CURR"] == input.client_id]
        if len(X) == 0:
            raise Exception(f"Client ID {input.client_id} doesn't exist")
        X = X.drop(["SK_ID_CURR"], axis=1)
        return X[input.client_infos].values.tolist()

    def model_features(self):
        return self.data.columns.tolist()
