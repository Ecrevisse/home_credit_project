import pandas as pd
import shap

import model as ml

model = ml.CreditModel()

X = model.data.drop(["SK_ID_CURR"], axis=1)
# change bool type to int
for col in X.columns:
    if X[col].dtype == "bool":
        X[col] = X[col].astype(int)

# print(X.describe())
# print(X.columns)
# print(X.isna().sum().sum())
explainer = shap.Explainer(model.model.predict, X[:1000])
shap_values = explainer(X[:1000])
print(shap_values)


shap.plots.waterfall(shap_values[0], max_display=14)
