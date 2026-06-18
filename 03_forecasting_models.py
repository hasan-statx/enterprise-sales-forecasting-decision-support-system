"""
03_forecasting_models.py
Multi-Model Sales Forecasting Pipeline
=========================================
Models: Linear Regression (feature-based), ARIMA, Prophet
Validation: Time-series train/test split, MAE / RMSE / MAPE

Author: Hasan Mahmud Sujan
"""
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import json
from datetime import datetime

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
import xgboost as xgb
from statsmodels.tsa.arima.model import ARIMA
from prophet import Prophet

BLUE   = "#1F4E79"
GREEN  = "#70AD47"
ORANGE = "#ED7D31"
GRAY   = "#555555"
COLORS = [BLUE, "#2E75B6", GREEN, ORANGE, "#7030A0"]

plt.rcParams.update({
    "font.size": 11, "axes.titleweight": "bold", "axes.titlecolor": BLUE,
    "axes.labelweight": "bold", "axes.labelcolor": BLUE,
    "figure.facecolor": "white", "axes.facecolor": "white",
    "axes.edgecolor": "#cccccc",
})

OUT = "/home/claude/sales_forecast/outputs"

print("=" * 65)
print("  ENTERPRISE SALES FORECASTING — MODEL TRAINING & VALIDATION")
print("=" * 65)

daily = pd.read_csv("/home/claude/sales_forecast/data/processed_sales_data.csv", parse_dates=["date"])
daily = daily.dropna().reset_index(drop=True)

# Hold out the final 90 days as test set
TEST_DAYS = 90
train = daily.iloc[:-TEST_DAYS].copy()
test  = daily.iloc[-TEST_DAYS:].copy()

print(f"\nTrain period: {train['date'].min().date()} -> {train['date'].max().date()}  ({len(train)} days)")
print(f"Test period : {test['date'].min().date()} -> {test['date'].max().date()}  ({len(test)} days)")

def evaluate(y_true, y_pred, name):
    mae  = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    print(f"  {name:20s} MAE={mae:>10,.0f}  RMSE={rmse:>10,.0f}  MAPE={mape:>6.2f}%")
    return {"model": name, "MAE": round(mae,1), "RMSE": round(rmse,1), "MAPE": round(mape,2)}

results = []

# ════════════════════════════════════════════════════════════════════════
# MODEL 1 — Linear Regression (feature-based)
# ════════════════════════════════════════════════════════════════════════
print("\n[1] Linear Regression (feature-based)")

FEATURES = ["lag_1","lag_7","lag_14","lag_28",
            "rolling_mean_7","rolling_mean_14","rolling_mean_30",
            "rolling_std_7","day_of_week","month","quarter",
            "is_weekend","is_construction_peak","is_post_monsoon_peak",
            "is_monsoon_dip","promo_share","pct_stores_open","trend_days"]

lr = LinearRegression()
lr.fit(train[FEATURES], train["sales"])
pred_lr = lr.predict(test[FEATURES])
results.append(evaluate(test["sales"].values, pred_lr, "Linear Regression"))

# ════════════════════════════════════════════════════════════════════════
# MODEL 2 — Random Forest
# ════════════════════════════════════════════════════════════════════════
print("\n[2] Random Forest")
rf = RandomForestRegressor(n_estimators=300, max_depth=8, min_samples_leaf=4,
                            random_state=42, n_jobs=-1)
rf.fit(train[FEATURES], train["sales"])
pred_rf = rf.predict(test[FEATURES])
results.append(evaluate(test["sales"].values, pred_rf, "Random Forest"))

# ════════════════════════════════════════════════════════════════════════
# MODEL 3 — XGBoost
# ════════════════════════════════════════════════════════════════════════
print("\n[3] XGBoost")
xgb_model = xgb.XGBRegressor(n_estimators=400, learning_rate=0.04, max_depth=5,
                              subsample=0.8, colsample_bytree=0.8,
                              random_state=42, verbosity=0)
xgb_model.fit(train[FEATURES], train["sales"])
pred_xgb = xgb_model.predict(test[FEATURES])
results.append(evaluate(test["sales"].values, pred_xgb, "XGBoost"))

# ════════════════════════════════════════════════════════════════════════
# MODEL 4 — ARIMA
# ════════════════════════════════════════════════════════════════════════
print("\n[4] ARIMA(5,1,2)")
arima_model = ARIMA(train["sales"].values, order=(5,1,2))
arima_fit = arima_model.fit()
pred_arima = arima_fit.forecast(steps=TEST_DAYS)
results.append(evaluate(test["sales"].values, pred_arima, "ARIMA(5,1,2)"))

# ════════════════════════════════════════════════════════════════════════
# MODEL 5 — Prophet
# ════════════════════════════════════════════════════════════════════════
print("\n[5] Prophet")
prophet_train = train[["date","sales"]].rename(columns={"date":"ds","sales":"y"})
m = Prophet(yearly_seasonality=True, weekly_seasonality=True,
            changepoint_prior_scale=0.1, seasonality_mode="multiplicative")
m.add_regressor("promo_share")
m.add_regressor("pct_stores_open")
prophet_train["promo_share"]     = train["promo_share"].values
prophet_train["pct_stores_open"] = train["pct_stores_open"].values
m.fit(prophet_train)

future = test[["date","promo_share","pct_stores_open"]].rename(columns={"date":"ds"})
fcst = m.predict(future)
pred_prophet = fcst["yhat"].values
results.append(evaluate(test["sales"].values, pred_prophet, "Prophet"))

# ════════════════════════════════════════════════════════════════════════
# MODEL COMPARISON TABLE
# ════════════════════════════════════════════════════════════════════════
results_df = pd.DataFrame(results).sort_values("MAPE")
print("\n" + "=" * 65)
print("  MODEL COMPARISON (sorted by MAPE)")
print("=" * 65)
print(results_df.to_string(index=False))

best_model_name = results_df.iloc[0]["model"]
print(f"\n  ✔ Best model: {best_model_name}")

results_df.to_csv(f"{OUT}/model_metrics.csv", index=False)

# ════════════════════════════════════════════════════════════════════════
# FORECAST VISUALISATION (best model = XGBoost typically wins on MAPE)
# ════════════════════════════════════════════════════════════════════════
pred_map = {
    "Linear Regression": pred_lr, "Random Forest": pred_rf,
    "XGBoost": pred_xgb, "ARIMA(5,1,2)": pred_arima, "Prophet": pred_prophet
}
best_pred = pred_map[best_model_name]

forecast_results = test[["date","sales"]].copy()
forecast_results["predicted"] = best_pred
forecast_results["model"] = best_model_name
forecast_results.to_csv(f"{OUT}/forecast_results.csv", index=False)

fig, axes = plt.subplots(2, 1, figsize=(13, 9), gridspec_kw={"height_ratios":[2,1]})

ax = axes[0]
ax.plot(train["date"].iloc[-180:], train["sales"].iloc[-180:],
        color=GRAY, lw=1.3, label="Historical (train)", alpha=0.7)
ax.plot(test["date"], test["sales"], color=BLUE, lw=2, label="Actual (test)")
ax.plot(test["date"], best_pred, color=ORANGE, lw=2, ls="--",
        label=f"Forecast ({best_model_name})")
ax.fill_between(test["date"], best_pred*0.92, best_pred*1.08,
                color=ORANGE, alpha=0.12, label="±8% band")
ax.axvline(test["date"].iloc[0], color="grey", ls=":", lw=1)
ax.set_title("Daily Sales Forecast vs Actual — Hold-out Test (90 days)")
ax.set_ylabel("Daily sales (BDT)")
ax.legend(loc="upper left", fontsize=9)
ax.grid(axis="y", color="#eeeeee")

ax2 = axes[1]
residuals = test["sales"].values - best_pred
ax2.bar(test["date"], residuals, color=[BLUE if r>=0 else "#C00000" for r in residuals], width=1)
ax2.axhline(0, color="grey", lw=1)
ax2.set_title("Forecast Residuals (Actual − Predicted)", fontsize=11)
ax2.set_ylabel("Residual (BDT)")
ax2.grid(axis="y", color="#eeeeee")

fig.tight_layout()
fig.savefig(f"{OUT}/forecast_plot.png", dpi=200, bbox_inches="tight")
plt.close(fig)
print(f"\n✔ Saved -> outputs/forecast_plot.png")

# Model comparison bar chart
fig2, ax = plt.subplots(figsize=(9,5.5))
results_sorted = results_df.sort_values("MAPE", ascending=True)
bars = ax.barh(results_sorted["model"], results_sorted["MAPE"],
               color=[ORANGE if m==best_model_name else "#9DC3E6" for m in results_sorted["model"]])
ax.set_xlabel("MAPE (%) — lower is better")
ax.set_title("Model Comparison: Forecast Accuracy (Hold-out MAPE)")
for bar, val in zip(bars, results_sorted["MAPE"]):
    ax.text(bar.get_width()+0.1, bar.get_y()+bar.get_height()/2, f"{val:.2f}%", va="center", fontsize=9)
ax.invert_yaxis()
ax.grid(axis="x", color="#eeeeee")
fig2.tight_layout()
fig2.savefig(f"{OUT}/model_comparison.png", dpi=200, bbox_inches="tight")
plt.close(fig2)
print(f"✔ Saved -> outputs/model_comparison.png")

# Save objects needed by next script
import joblib
joblib.dump({"model_name": best_model_name, "features": FEATURES}, f"{OUT}/best_model_meta.pkl")
if best_model_name == "XGBoost":
    joblib.dump(xgb_model, f"{OUT}/best_model.pkl")
elif best_model_name == "Random Forest":
    joblib.dump(rf, f"{OUT}/best_model.pkl")
else:
    joblib.dump(lr, f"{OUT}/best_model.pkl")

print("\n✔ Pipeline complete.")
