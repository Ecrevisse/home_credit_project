import pandas as pd
import shap
import numpy as np
import model as ml
import matplotlib.pyplot as plt

model = ml.CreditModel()

X = model.valid_data.drop(["SK_ID_CURR"], axis=1)
# change bool type to int
for col in X.columns:
    if X[col].dtype == "bool":
        X[col] = X[col].astype(int)

# print(X.describe())
# print(X.columns)
# print(X.isna().sum().sum())
shap_values = model.explainer(X)  # [:1000])
res = model.shap_global()

shap_val_local = res["shap_values"]
base_value = res["base_value"]
feat_values = res["data"]

# explanation = shap.Explanation(
#     np.reshape(np.array(shap_val_local, dtype="float"), (1, -1)),
#     base_value,
#     data=np.reshape(np.array(feat_values, dtype="float"), (1, -1)),
#     feature_names=feat_names,
# )

explanation = shap.Explanation(
    np.array(shap_val_local),
    np.array(base_value),
    data=np.array(feat_values),
)
# print(shap_values)
# print(shap_values[:, :, 1])
print(explanation)
shap.plots.beeswarm(shap_values[:, :, 1], max_display=10, show=False)
# we retrieve le figure from shap
fig = plt.gcf()
print(type(fig))
fig.show()
# we save the figure in a file
fig.savefig("test.png", format="png", bbox_inches="tight")
# shap.plots.waterfall(shap_values[0], max_display=14)

# x_test = X[:100]
# y_test = model.model.predict(x_test)

# from explainerdashboard import ClassifierExplainer, ExplainerDashboard

# exp = ClassifierExplainer(model.model, x_test, y_test)

# from explainerdashboard.custom import *


# class CustomDashboard(ExplainerComponent):
#     def __init__(self, explainer, title="Custom Dashboard", name="None"):
#         super().__init__(explainer, title, name=name)
#         self.shap_dependence = ShapSummaryComponent(
#             exp,
#             depth=15,
#             summary_type="detailed",
#         )

#     def layout(self):
#         return html.Div([self.shap_dependence.layout()])


# ExplainerDashboard(exp, CustomDashboard).run()

# # db = ExplainerDashboard(exp)
# # db.run(port=8050)
