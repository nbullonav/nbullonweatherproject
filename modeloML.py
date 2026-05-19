import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sb
# Cargar Librerias para tratamiento de datos
import missingno as mno
import random
import os
from math import sqrt
from datetime import datetime
from statsforecast import StatsForecast
from statsforecast.models import AutoARIMA
from statsmodels.tsa.arima.model import ARIMA
from statsforecast.arima import ARIMASummary
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.model_selection import TimeSeriesSplit, RandomizedSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error
from xgboost import XGBRegressor
import statsmodels.api as sm
import statsmodels.api as sm
import os

# ================== 1. cargar datos ==================
station_id = os.environ["STATION_ID"]
csv_path = f"data/station{station_id}.csv"

df = pd.read_csv(csv_path, parse_dates=["time"])
df = df.set_index("time").sort_index()

df = df.drop(columns=["NINO12","NINO34"])
df["log_prcp"] = np.log1p(df["prcp"])

# ================== 2. Lags (OK antes del split) ==================
for k in [1,2,3,6,12,]:
    df[f"prcp_lag{k}"] = df["log_prcp"].shift(k)

for k in range(1,13):
    df[f"ANOM34_lag{k}"] = df["NINO34ANOM"].shift(k)

for k in [1,3]:
    df[f"ICEN_lag{k}"] = df["ICEN"].shift(k)

# ================== 3. X month_sin y month_cos ================== 
df["month"] = df.index.month
#df = pd.get_dummies(df, columns=["month"], prefix="month", drop_first=False)
df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)

# ================== 4. Split base ==================
split_idx = int(len(df) * 0.8)
df_train = df.iloc[:split_idx].copy()   # 1982-2017 (428)
df_test  = df.iloc[split_idx:].copy()   # 2017-2026 (107)

def add_rollings_precenso(d):
    d["prcp_roll3"]  = d["log_prcp"].rolling(3).mean()
    d["prcp_roll6"]  = d["log_prcp"].rolling(6).mean()
    d["prcp_roll12"] = d["log_prcp"].rolling(12).mean()
    d["prcp_std6"]   = d["log_prcp"].rolling(6).std()
    d["enso34_roll3"]  = d["NINO34ANOM"].rolling(3).mean()
    d["ICEN_roll3"] = d["ICEN"].rolling(3).mean()

add_rollings_precenso(df_train)
add_rollings_precenso(df_test)

df_train_pres = df_train.copy()
df_test_pres  = df_test.copy()

# ================== 6. Concatenar ==================
df_model = pd.concat([df_train, df_test])


# ================== 7. Definir grupos de columnas ==================
lag_features = (
    [f"prcp_lag{k}" for k in [1,2,3,6,12]] +
    # [f"ANOM12_lag{k}" for k in range(1,13)] +
    [f"ANOM34_lag{k}" for k in range(1,13)] +
    [f"ICEN_lag{k}" for k in [1,3]]
)

rolling_features = [
    "prcp_roll3", "prcp_roll6", "prcp_roll12","prcp_std6", #"prcp_std12",
    #"enso12_roll3", "enso12_roll6",
    "enso34_roll3", #"enso34_roll6",
    "ICEN_roll3"#,"ICEN_roll6"
]

# month_features = [col for col in df.columns if col.startswith("month_")]
month_features = ["month_sin", "month_cos"]

feature_cols = lag_features + rolling_features + month_features

exog_sarimax = ["NINO34ANOM","ICEN"]

# ================== 8. Armar df_model limpio (XGB) ==================
df_model = df_model[["log_prcp", "ICEN","NINO34ANOM"] + feature_cols].dropna()

# agregar nuevo split_idx basado en df_model ya sin NaN
split_idx = int(len(df_model) * 0.8)

y = df_model["log_prcp"]
X_xgb      = df_model[feature_cols]

y_train = y.iloc[:split_idx]
y_test  = y.iloc[split_idx:]

true_train = np.expm1(y_train)
true_test  = np.expm1(y_test)

X_train = X_xgb.iloc[:split_idx]
X_test = X_xgb.iloc[split_idx:]

# ================== 9. Exógenas para SARIMAX ==================
exog_train = df_model[exog_sarimax].iloc[:split_idx]   # sarimax 1 var. exógena
exog_test  = df_model[exog_sarimax].iloc[split_idx:]

# ---------------------------------
# ----- Funciones auxiliares ------
# ---------------------------------
def rmse(y_true, y_pred):
    return sqrt(mean_squared_error(y_true, y_pred))

def mae(y_true, y_pred):
    return mean_absolute_error(y_true, y_pred)

# array de resultados
resultados = []

####### MODELO GANADOR #######
import statsmodels.api as sm

exog_sarimax = ['NINO34ANOM',"ANOM34_lag1"]
df_sarimax = df_model[["log_prcp"] + exog_sarimax].dropna()

split_idx_sarimax = int(len(df_sarimax) * 0.8)

y_sarimax = df_sarimax["log_prcp"]
exog_sarimax_full = df_sarimax[exog_sarimax]

y_train_sarimax = y_sarimax.iloc[:split_idx_sarimax]
y_test_sarimax  = y_sarimax.iloc[split_idx_sarimax:]

exog_train_sarimax = exog_sarimax_full.iloc[:split_idx_sarimax]
exog_test_sarimax  = exog_sarimax_full.iloc[split_idx_sarimax:]

sarimax_model = sm.tsa.statespace.SARIMAX(
    y_train_sarimax,
    exog=exog_train_sarimax,
    order=(2,0,2),
    seasonal_order=(1,1,1,12),
    enforce_stationarity=False,
    enforce_invertibility=False
).fit(disp=False)

sarimaxhybrid2_train_pred = hybrid2_model.predict(start=y_train.index[0], end=y_train.index[-1], exog=exog_train_sarimax)

start = len(y_train)
end = len(y_train) + len(y_test) - 1

sarimax_train_pred = sarimax_model.predict(start=y_train.index[0], end=y_train.index[-1], exog=exog_train_sarimax)

start = len(y_train)
end = len(y_train) + len(y_test) - 1

sarimax_test_pred = sarimax_model.predict(start=start, end=end, exog=exog_test_sarimax)
sarimax_test_pred_series = pd.Series(sarimax_test_pred.values, index=y_test.index)

residuals_train = y_train - sarimax_train_pred

sarimax_train_real = np.expm1(sarimax_train_pred)
sarimax_test_real  = np.expm1(sarimax_test_pred)

true_train_sarimax = np.expm1(y_train_sarimax)
true_test_sarimax  = np.expm1(y_test_sarimax)

rmse_sarimax = rmse(true_test_sarimax, sarimax_test_real)
mae_sarimax  = mae(true_test_sarimax, sarimax_test_real)

resultados.append({
    "modelo": "SARIMAX (1,1,1) (1,1,1,12) ENSO rezagado",
    "RMSE": rmse_sarimax,
    "MAE": mae_sarimax
})


# XGBOOST
randomstate = 42
tscv = TimeSeriesSplit(n_splits=5)

param_grid = {
    "n_estimators": [200, 300, 400, 500],
    "max_depth": [2, 3, 4, 5],
    "learning_rate": [0.01, 0.03, 0.05, 0.1],
    "subsample": [0.6, 0.8, 1.0],
    "colsample_bytree": [0.6, 0.8, 1.0],
    "gamma": [0, 1, 5],
    "min_child_weight": [1, 3, 5],
}

xgb_model = XGBRegressor(objective="reg:squarederror", random_state=randomstate)

search = RandomizedSearchCV(
    estimator=xgb_model,
    param_distributions=param_grid,
    n_iter=40,
    scoring="neg_mean_squared_error",
    cv=tscv,
    verbose=1,
    n_jobs=-1
)

search.fit(X_train.to_numpy(), residuals_train.to_numpy())
best_xgb = search.best_estimator_


xgb_train_pred = best_xgb.predict(X_train)
xgb_test_pred  = best_xgb.predict(X_test)
xgb_test_pred_series = pd.Series(xgb_test_pred, index=X_test.index)

xgb_train_real = np.expm1(xgb_train_pred)
xgb_test_real  = np.expm1(xgb_test_pred)

hybrid2_train = true_train_sarimax + xgb_train_real
hybrid2_test  = true_test_sarimax  + xgb_test_real


# Prediccion futura

# ================================
# 1. Crear df_future base
# ================================
H = 12

future_index = pd.date_range(
    start=df_model.index[-1] + pd.offsets.MonthBegin(1),
    periods=H,
    freq="MS"
)

df_future = pd.DataFrame(index=future_index)

# ENSO futuro

# ejemplo Niño moderado
#df_future["ANOM"] = [0.10, 0.23, 0.9, 1.1, 1.2, 1.0, 0.8, 0.6, 0.4, 0.3, 0.2, 0.1]
# ejemplo nino fuerte
df_future["NINO34ANOM"] = [0.10, 0.23, 1.4, 1.7, 1.9, 1.8, 1.6, 1.3, 1.0, 0.8, 0.6, 0.4]
# ejemplo nina moderada
# df_future["ANOM"] = [0.1, 0.0, -0.2, -0.4, -0.6, -0.8, -1.0, -1.1, -1.0, -0.8, -0.6, -0.4]

# ================================
# 2. Extender df original
# ================================
df_ext = pd.concat([df, df_future], axis=0)

# ================================
# 3. Month dummies X   month_sin y month_cos
# ================================
# ================== 3. Dummies de mes ==================
df_ext["month"] = df_ext.index.month
df_ext["month_sin"] = np.sin(2 * np.pi * df_ext["month"] / 12)
df_ext["month_cos"] = np.cos(2 * np.pi * df_ext["month"] / 12)

# ================================
# 4. Regenerar TODAS las features en df_ext
# ================================

# Lags prcp
for k in [1,2,3,6,12]:
    df_ext[f"prcp_lag{k}"] = df_ext["log_prcp"].shift(k)

# Lags ENSO
for k in range(1,13):
    df_ext[f"ANOM34_lag{k}"] = df_ext["NINO34ANOM"].shift(k)

for k in [1,3]:
    df_ext[f"ICEN_lag{k}"] = df_ext["ICEN"].shift(k)

# Rolling prcp
df_ext["prcp_roll3"]  = df_ext["log_prcp"].rolling(3).mean()
df_ext["prcp_roll6"]  = df_ext["log_prcp"].rolling(6).mean()
df_ext["prcp_roll12"] = df_ext["log_prcp"].rolling(12).mean()

df_ext["prcp_std6"]   = df_ext["log_prcp"].rolling(6).std()

# Rolling ENSO
df_ext["enso34_roll3"] = df_ext["NINO34ANOM"].rolling(3).mean()
df_ext["ICEN_roll3"] = df_ext["ICEN"].rolling(3).mean()

# ================================
# 5. Seleccionar exógenas SARIMAX
# ================================

exog_sarimax = ["ICEN_lag1","NINO34ANOM"]
exog_future = df_ext.loc[future_index, exog_sarimax].fillna(0)

# hibrid #1
#exog_sarimax = list(exog_train.columns)
#exog_future = df_ext.loc[future_index, exog_sarimax].fillna(0)

# ================================
# 6. Seleccionar features XGB
# ================================
X_future = df_ext.loc[future_index, X_train.columns].fillna(0)

# ================================
# 7. Forecast SARIMAX
# ================================
start = sarimax_model.nobs
end   = sarimax_model.nobs + H - 1

sarimax2_forecast = sarimax_model.predict(start=start, end=end, exog=exog_future)
sarimax2_forecast = np.array(sarimax2_forecast)
sarimax2_future_series = pd.Series(sarimax2_forecast, index=future_index)

# ================================
# 8. Forecast XGB futuro
# ================================
xgb_future_pred = best_xgb.predict(X_future.to_numpy())
xgb_future_series = pd.Series(xgb_future_pred, index=future_index)

# ================================
# 9. Modelo híbrido
# ================================
hybrid_future_series = sarimax2_future_series + xgb_future_series

hybrid_train = sarimax_train_real + xgb_train_real
hybrid_test  = sarimax_test_real  + xgb_test_real

# metricas RMSE & MAE
resultados.append({
    "modelo": "Híbrido (SARIMAX + XGB)",
    "RMSE": rmse(true_test_sarimax, hybrid_test),
    "MAE": mae(true_test_sarimax, hybrid_test)
})

dhistoric = y.to_frame("historic").reset_index().rename(columns={"index": "time"})
dsarimax = sarimax2_future_series.to_frame("sarimax").reset_index().rename(columns={"index": "time"})
dxgb = xgb_future_series.to_frame("xgb").reset_index().rename(columns={"index": "time"})
dhybrid = hybrid_future_series.to_frame("hybrid").reset_index().rename(columns={"index": "time"})

df_final = (
    dhistoric
    .merge(dsarimax, on="time", how="outer")
    .merge(dxgb, on="time", how="outer")
    .merge(dhybrid, on="time", how="outer")
    .sort_values("time")
)
csvfilefinal = "data/pred_" + station_id + ".csv"
df_final.to_csv(csvfilefinal, index=False)

test_df = pd.DataFrame({
    "time": true_test_sarimax.index,
    "real": true_test_sarimax.values,
    "sarimax": sarimax_test_pred_series.values,
    "xgb": xgb_test_pred_series.values,
    "hybrid": hybrid_test.values
})

test_df.set_index("time", inplace=True)
csvfiletest = "data/test_" + station_id + ".csv"
test_df.to_csv(csvfiletest)
