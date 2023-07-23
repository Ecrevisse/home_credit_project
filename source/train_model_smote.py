import numpy as np
import pandas as pd
import re
import gc
import time
import sys
from contextlib import contextmanager
from lightgbm import LGBMClassifier
from lightgbm import early_stopping
from lightgbm import log_evaluation
from sklearn.metrics import (
    roc_auc_score,
    roc_curve,
    RocCurveDisplay,
    confusion_matrix,
    ConfusionMatrixDisplay,
)
from sklearn.model_selection import KFold, StratifiedKFold, train_test_split
from sklearn.feature_selection import SelectFromModel
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from urllib.parse import urlparse

import mlflow
from mlflow.models import infer_signature
import mlflow.sklearn

from imblearn.over_sampling import SMOTE

import logging

logging.basicConfig(level=logging.WARN)
logger = logging.getLogger(__name__)

warnings.simplefilter(action="ignore", category=FutureWarning)


# feature selection
def select_features(X_train, X_test, count_cols=None):
    if count_cols is None:
        return X_train, X_test, X_train.columns
    fi_df = pd.read_csv("feature_importance.csv")
    cols = (
        fi_df[["feature", "importance"]]
        .groupby("feature")
        .mean()
        .sort_values(by="importance", ascending=False)[:count_cols]
        .index
    )
    X_train_fs = X_train[cols]
    X_test_fs = X_test[cols]
    return X_train_fs, X_test_fs, cols


def get_confusion_matrix(model, X_test, y_test, threshold=0.5):
    df = pd.DataFrame(model.predict_proba(X_test))
    predictions = np.where(df.iloc[:, 1] > threshold, 1, 0)
    cm = confusion_matrix(y_test, predictions, labels=model.classes_)
    return cm


def eval_metrics(actual, pred, model, X_test, y_test, threshold=0.5):
    auc = roc_auc_score(actual, pred)
    cm = get_confusion_matrix(model, X_test, y_test, threshold)
    TN = cm[0][0]
    FP = cm[0][1] * 10  # FP is 10 times worst than FN
    FN = cm[1][0]
    TP = cm[1][1]
    F1_score = 2 * TP / (2 * TP + FP + FN)  # F1 score
    return auc, F1_score


# Display roc curve
def display_roc_curve(model, X_test, y_test):
    RocCurveDisplay.from_estimator(model, X_test, y_test)
    plt.savefig("roc_curve.png")


def display_confusion_matrix(model, X_test, y_test, threshold=0.5):
    cm = get_confusion_matrix(model, X_test, y_test, threshold)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=model.classes_)
    disp.plot()
    plt.savefig("confusion_matrix.png")


# Main code
if __name__ == "__main__":
    df = pd.read_csv("../input/cleaned_data.csv", index_col="index")

    # Divide in training/validation and test data
    train_df = df[df["TARGET"].notnull()]
    test_df = df[df["TARGET"].isnull()]
    print(
        "Starting LightGBM. Train shape: {}, test shape: {}".format(
            train_df.shape, test_df.shape
        )
    )
    gc.collect()
    # Cross validation model

    # some clean
    train_df = train_df.fillna(0)
    train_df = train_df.replace([np.inf, -np.inf], 0)

    count_cols = int(sys.argv[1]) if len(sys.argv) > 1 else None
    count_rows = int(sys.argv[2]) if len(sys.argv) > 2 else None
    threshold = float(sys.argv[3]) if len(sys.argv) > 3 else None

    if count_rows is not None:
        train_df_0 = train_df[train_df["TARGET"] == 0]
        train_df_1 = train_df[train_df["TARGET"] == 1]

        train_df_0 = train_df_0.sample(
            max(count_rows, len(train_df_1)), random_state=42
        )

        train_df = pd.concat([train_df_0, train_df_1])

    feats = [
        f
        for f in train_df.columns
        if f not in ["SK_ID_CURR", "SK_ID_BUREAU", "SK_ID_PREV", "index"]
    ]
    train_df = train_df[feats]
    # test_df = test_df[feats]

    train_x, valid_x, train_y, valid_y = train_test_split(
        train_df.drop(["TARGET"], axis=1),
        train_df["TARGET"],
        stratify=train_df["TARGET"],
        random_state=0,
    )

    train_x, valid_x, cols = select_features(train_x, valid_x, count_cols)

    oversample = SMOTE()
    train_x, train_y = oversample.fit_resample(train_x, train_y)

    train_x = train_x.rename(columns=lambda x: re.sub("[^A-Za-z0-9_]+", "", x))
    valid_x = valid_x.rename(columns=lambda x: re.sub("[^A-Za-z0-9_]+", "", x))

    train_y_np = np.array(train_y)
    train_0_count = len(train_y_np[np.where(train_y_np == 0)])
    train_1_count = len(train_y_np[np.where(train_y_np == 1)])

    with mlflow.start_run():
        # LightGBM parameters found by Bayesian optimization
        clf = LGBMClassifier(
            n_estimators=10000,
            learning_rate=0.02,
            num_leaves=34,
            colsample_bytree=0.9497036,
            subsample=0.8715623,
            max_depth=8,
            reg_alpha=0.041545473,
            reg_lambda=0.0735294,
            min_split_gain=0.0222415,
            min_child_weight=39.3259775,
            verbose=-1,
        )

        clf.fit(
            train_x,
            train_y,
            eval_set=[(train_x, train_y), (valid_x, valid_y)],
            eval_metric="auc",
            callbacks=[early_stopping(200), log_evaluation(200)],
        )

        pred_y = clf.predict_proba(valid_x, num_iteration=clf.best_iteration_)[:, 1]

        auc, F1_score = eval_metrics(valid_y, pred_y, clf, valid_x, valid_y, threshold)
        display_roc_curve(clf, valid_x, valid_y)
        display_confusion_matrix(clf, valid_x, valid_y, threshold)

        mlflow.log_param("model", "LightGBM")
        mlflow.log_param(
            "count_cols", count_cols if count_cols is not None else len(train_x.columns)
        )

        mlflow.log_param("train_0_count", train_0_count)
        mlflow.log_param("train_1_count", train_1_count)
        mlflow.log_param("0_1_ratio", (train_0_count / (train_0_count + train_1_count)))
        mlflow.log_param("threshold", threshold)
        mlflow.log_metric("auc", auc)
        mlflow.log_metric("F1_score", F1_score)

        pred = clf.predict_proba(train_x, num_iteration=clf.best_iteration_)[:, 1]
        signature = infer_signature(train_x, pred)

        tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme

        # Model registry does not work with file store
        if tracking_url_type_store != "file":
            # Register the model
            # There are other ways to use the Model Registry, which depends on the use case,
            # please refer to the doc for more information:
            # https://mlflow.org/docs/latest/model-registry.html#api-workflow
            mlflow.sklearn.log_model(
                clf,
                "model",
                registered_model_name="LightGBMHomeCredit",
                signature=signature,
            )
        else:
            mlflow.sklearn.log_model(clf, "model", signature=signature)

        del clf, train_x, train_y, valid_x, valid_y
        gc.collect()
