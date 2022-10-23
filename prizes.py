"""Prizes modelling."""

# %%

import numpy as np
import pandas as pd
import plotly.express as px
from scipy.optimize import curve_fit

QUERY = "SELECT * FROM express.fct_norm_points WHERE season >= 2022 AND round >= 20"
data = pd.read_gbq(QUERY)

px.line(data, x="points_norm", y="prizes", color="round", log_y=False)

# %%

ROUND = 23

def func(x, a, b, c):
    return a * np.exp(b * x) + c

X = data[data["round"] == ROUND]["points_norm"]
y = data[data["round"] == ROUND]["prizes"]
X_pred = np.linspace(0.0, 0.6, endpoint=True)

popt, _ = curve_fit(func, X, y)
y_pred = func(X_pred, *popt)

pred = pd.DataFrame()
pred["points_norm"] = X_pred
pred["prizes"] = y_pred
pred["round"] = "model"

data_plot = data[data["round"].isin((ROUND, "model"))]
px.line(pd.concat((data_plot, pred)), x="points_norm", y="prizes", color="round", log_y=False)


# %%

import lightgbm as lgbm

X = data[["points_norm"]]
y = data["prizes"]
X_pred = np.linspace(0.0, 0.6, endpoint=True)

model = lgbm.LGBMRegressor(objective="poisson")
model.fit(X, y)
y_pred = model.predict(X_pred.reshape(-1, 1))

pred = pd.DataFrame()
pred["points_norm"] = X_pred
pred["prizes"] = y_pred
pred["round"] = "model"

px.line(pd.concat((data, pred)), x="points_norm", y="prizes", color="round", log_y=False)
