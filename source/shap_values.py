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
# explainer = shap.Explainer(model.model.predict, X[:1000])
# shap_values = explainer(X[:10])
# print(shap_values)


# shap.plots.waterfall(shap_values[0], max_display=14)

x_test = X[:100]
y_test = model.model.predict(x_test)

from explainerdashboard import ClassifierExplainer, ExplainerDashboard

exp = ClassifierExplainer(model.model, x_test, y_test)

from explainerdashboard.custom import *


class CustomDashboard(ExplainerComponent):
    def __init__(self, explainer, title="Custom Dashboard", name="None"):
        super().__init__(explainer, title, name=name)
        self.shap_dependence = ShapSummaryComponent(
            exp,
            depth=15,
            summary_type="detailed",
        )

    def layout(self):
        return html.Div([self.shap_dependence.layout()])


ExplainerDashboard(exp, CustomDashboard).run()

# db = ExplainerDashboard(exp)
# db.run(port=8050)
