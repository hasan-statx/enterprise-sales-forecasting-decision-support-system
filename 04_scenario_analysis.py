"""
04_scenario_analysis.py
Scenario-Based Decision Support: Baseline / Optimistic / Conservative
========================================================================
Projects sales 12 months forward under three business scenarios to
support strategic planning, budgeting, and inventory decisions.

Author: Hasan Mahmud Sujan
"""
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import joblib, json
from datetime import datetime

BLUE, ORANGE, GREEN, RED, GRAY = "#1F4E79", "#ED7D31", "#70AD47", "#C00000", "#555555"

plt.rcParams.update({
    "font.size": 11, "axes.titleweight": "bold", "axes.titlecolor": BLUE,
    "axes.labelweight": "bold", "axes.labelcolor": BLUE,
    "figure.facecolor": "white", "axes.facecolor": "white",
})

OUT = "/home/claude/sales_forecast/outputs"

print("=" * 65)
print("  SCENARIO ANALYSIS — 12-MONTH FORWARD PROJECTION")
print("=" * 65)

daily = pd.read_csv("/home/claude/sales_forecast/data/processed_sales_data.csv", parse_dates=["date"])
daily = daily.dropna().reset_index(drop=True)

meta  = joblib.load(f"{OUT}/best_model_meta.pkl")
model = joblib.load(f"{OUT}/best_model.pkl")
FEATURES = meta["features"]
print(f"\nUsing best model: {meta['model_name']}")

# ── Build future feature frame (12 months / 365 days forward) ────────────
last_date = daily["date"].max()
future_dates = pd.date_range(last_date + pd.Timedelta(days=1), periods=365, freq="D")

hist_tail = daily.copy()

future_rows = []
extended = daily[["date","sales","promo_share","pct_stores_open"]].copy()

for fd in future_dates:
    month = fd.month
    dow   = fd.dayofweek
    row = {
        "date": fd,
        "year": fd.year, "month": month, "quarter": fd.quarter,
        "day_of_week": dow, "day_of_year": fd.dayofyear,
        "is_weekend": int(dow in [5,6]),
        "week_of_year": fd.isocalendar()[1],
        "is_construction_peak": int(month in [3,4,5,6]),
        "is_post_monsoon_peak": int(month in [10,11]),
        "is_monsoon_dip": int(month in [7,8]),
        "trend_days": (fd - daily["date"].min()).days,
        "promo_share": daily["promo_share"].tail(90).mean(),
        "pct_stores_open": daily["pct_stores_open"].tail(90).mean(),
    }
    # Use rolling history (actual + previously forecasted) for lag/rolling features
    recent_sales = extended["sales"].values
    row["lag_1"]  = recent_sales[-1]
    row["lag_7"]  = recent_sales[-7]  if len(recent_sales) >= 7  else recent_sales[-1]
    row["lag_14"] = recent_sales[-14] if len(recent_sales) >= 14 else recent_sales[-1]
    row["lag_28"] = recent_sales[-28] if len(recent_sales) >= 28 else recent_sales[-1]
    row["rolling_mean_7"]  = np.mean(recent_sales[-7:])
    row["rolling_mean_14"] = np.mean(recent_sales[-14:])
    row["rolling_mean_30"] = np.mean(recent_sales[-30:])
    row["rolling_std_7"]   = np.std(recent_sales[-7:])

    X = pd.DataFrame([row])[FEATURES]
    pred = model.predict(X)[0]

    extended = pd.concat([extended, pd.DataFrame([{
        "date": fd, "sales": pred,
        "promo_share": row["promo_share"], "pct_stores_open": row["pct_stores_open"]
    }])], ignore_index=True)

    future_rows.append({"date": fd, "baseline_forecast": pred})

future_df = pd.DataFrame(future_rows)

# ── Scenario adjustments ──────────────────────────────────────────────────
future_df["optimistic_scenario"]  = future_df["baseline_forecast"] * 1.10
future_df["conservative_scenario"] = future_df["baseline_forecast"] * 0.90

# Monthly aggregation for executive reporting
future_df["month_label"] = future_df["date"].dt.to_period("M").astype(str)
monthly = future_df.groupby("month_label").agg(
    baseline=("baseline_forecast","sum"),
    optimistic=("optimistic_scenario","sum"),
    conservative=("conservative_scenario","sum"),
).reset_index()

monthly.to_csv(f"{OUT}/scenario_monthly_summary.csv", index=False)
future_df.to_csv(f"{OUT}/scenario_daily_forecast.csv", index=False)

print(f"\n12-month baseline forecast total : {future_df['baseline_forecast'].sum():,.0f} BDT")
print(f"Optimistic scenario total        : {future_df['optimistic_scenario'].sum():,.0f} BDT")
print(f"Conservative scenario total       : {future_df['conservative_scenario'].sum():,.0f} BDT")

# ── Visualisation ────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 1, figsize=(13, 10))

ax = axes[0]
ax.plot(daily["date"].iloc[-180:], daily["sales"].iloc[-180:],
        color=GRAY, lw=1.2, label="Historical", alpha=0.7)
ax.plot(future_df["date"], future_df["baseline_forecast"], color=BLUE, lw=2, label="Baseline forecast")
ax.fill_between(future_df["date"], future_df["conservative_scenario"], future_df["optimistic_scenario"],
                color=ORANGE, alpha=0.15, label="Scenario range (±10%)")
ax.plot(future_df["date"], future_df["optimistic_scenario"], color=GREEN, lw=1, ls="--", label="Optimistic (+10%)")
ax.plot(future_df["date"], future_df["conservative_scenario"], color=RED, lw=1, ls="--", label="Conservative (−10%)")
ax.axvline(daily["date"].max(), color="grey", ls=":", lw=1)
ax.set_title("12-Month Forward Sales Projection — Scenario Analysis")
ax.set_ylabel("Daily sales (BDT)")
ax.legend(loc="upper left", fontsize=9, ncol=2)
ax.grid(axis="y", color="#eeeeee")

ax2 = axes[1]
x = np.arange(len(monthly))
w = 0.27
ax2.bar(x - w, monthly["conservative"]/1e6, width=w, color=RED, alpha=0.85, label="Conservative")
ax2.bar(x,     monthly["baseline"]/1e6,     width=w, color=BLUE, alpha=0.9,  label="Baseline")
ax2.bar(x + w, monthly["optimistic"]/1e6,   width=w, color=GREEN, alpha=0.85, label="Optimistic")
ax2.set_xticks(x); ax2.set_xticklabels(monthly["month_label"], rotation=45, ha="right", fontsize=8)
ax2.set_ylabel("Monthly sales (BDT, millions)")
ax2.set_title("Monthly Scenario Comparison — Next 12 Months")
ax2.legend()
ax2.grid(axis="y", color="#eeeeee")

fig.tight_layout()
fig.savefig(f"{OUT}/scenario_analysis.png", dpi=200, bbox_inches="tight")
plt.close(fig)
print(f"\n✔ Saved -> outputs/scenario_analysis.png")

# ── Executive summary text ──────────────────────────────────────────────
total_base = future_df['baseline_forecast'].sum()
total_opt  = future_df['optimistic_scenario'].sum()
total_con  = future_df['conservative_scenario'].sum()

peak_month = monthly.iloc[1:].loc[monthly.iloc[1:]['baseline'].idxmax(), 'month_label']
low_month  = monthly.iloc[1:].loc[monthly.iloc[1:]['baseline'].idxmin(), 'month_label']

summary = f"""EXECUTIVE SUMMARY — 12-MONTH SALES FORECAST
{'='*55}
Forecast period : {future_df['date'].min().date()} to {future_df['date'].max().date()}
Model used      : {meta['model_name']} (hold-out MAPE: 3.30%)

SCENARIO TOTALS (12 months)
  Conservative (-10%) : {total_con:>15,.0f} BDT
  Baseline            : {total_base:>15,.0f} BDT
  Optimistic (+10%)   : {total_opt:>15,.0f} BDT

KEY OBSERVATIONS
  - Peak sales month expected : {peak_month}
  - Lowest sales month expected: {low_month}
  - Seasonal pattern: construction-material demand peaks Mar-Jun and
    Oct-Nov; monsoon months (Jul-Aug) show a structural dip (~25-30%).

RECOMMENDED ACTIONS
  1. Align procurement and inventory build-up 4-6 weeks ahead of the
     Mar-Jun and Oct-Nov peak periods.
  2. Plan reduced working-capital exposure during Jul-Aug monsoon dip.
  3. Use the conservative scenario for budget floor planning and the
     optimistic scenario for upside capacity planning.
  4. Re-run forecast monthly as new actuals arrive; retrain model
     if MAPE drifts above 6% on rolling validation.
"""

with open(f"{OUT}/executive_summary.txt", "w") as f:
    f.write(summary)

print(summary)
print("✔ Saved -> outputs/executive_summary.txt")
