"""
05_executive_dashboard.py
Power BI-style Executive Dashboard Mockup + Model Documentation
====================================================================
Generates a static dashboard image (KPI cards + 4 visuals) representing
what a live Power BI dashboard would show management, plus a
structured model_card.json for governance documentation.

Author: Hasan Mahmud Sujan
"""
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
import json
from datetime import datetime

BLUE, ORANGE, GREEN, RED, GRAY, LIGHT = "#1F4E79","#ED7D31","#70AD47","#C00000","#555555","#EBF3FA"

plt.rcParams.update({"font.size": 10, "figure.facecolor":"white"})

OUT = "/home/claude/sales_forecast/outputs"
DASH = "/home/claude/sales_forecast/dashboard"

daily   = pd.read_csv("/home/claude/sales_forecast/data/processed_sales_data.csv", parse_dates=["date"])
future  = pd.read_csv(f"{OUT}/scenario_daily_forecast.csv", parse_dates=["date"])
monthly = pd.read_csv(f"{OUT}/scenario_monthly_summary.csv")
metrics = pd.read_csv(f"{OUT}/model_metrics.csv")

# ════════════════════════════════════════════════════════════════════════
# DASHBOARD LAYOUT — 4 KPI cards + 4 visuals
# ════════════════════════════════════════════════════════════════════════
fig = plt.figure(figsize=(18, 11))
fig.patch.set_facecolor("white")
gs = gridspec.GridSpec(3, 4, figure=fig, height_ratios=[0.5,1.3,1.3],
                        hspace=0.55, wspace=0.35)

# ── Title bar ────────────────────────────────────────────────────────────
fig.text(0.02, 0.975, "ENTERPRISE SALES — EXECUTIVE DASHBOARD", fontsize=18,
          fontweight="bold", color=BLUE)
fig.text(0.02, 0.955, f"Data refreshed: {daily['date'].max().date()}  |  Forecast horizon: 12 months  |  Model: XGBoost (MAPE 3.30%)",
          fontsize=10, color=GRAY)

# ── KPI cards (row 0) ────────────────────────────────────────────────────
yoy_growth = ((daily['sales'].tail(365).sum() / daily['sales'].iloc[-730:-365].sum()) - 1) * 100
kpi_specs = [
    ("Total Sales (Trailing 12M)", f"{daily['sales'].tail(365).sum()/1e6:,.1f}M BDT", BLUE),
    ("Forecast Accuracy (MAPE)",   f"{metrics.iloc[0]['MAPE']:.2f}%", GREEN),
    ("12M Baseline Forecast",      f"{monthly['baseline'].sum()/1e6:,.1f}M BDT", ORANGE),
    ("YoY Growth Rate",            f"+{yoy_growth:.1f}%", "#7030A0"),
]
for i,(label, value, color) in enumerate(kpi_specs):
    ax = fig.add_subplot(gs[0, i])
    ax.axis("off")
    rect = mpatches.FancyBboxPatch((0.02,0.05), 0.96, 0.9, boxstyle="round,pad=0.02",
                                    linewidth=1.5, edgecolor=color, facecolor=LIGHT, transform=ax.transAxes)
    ax.add_patch(rect)
    ax.text(0.5, 0.62, value, ha="center", va="center", fontsize=20, fontweight="bold", color=color, transform=ax.transAxes)
    ax.text(0.5, 0.22, label, ha="center", va="center", fontsize=9.5, color=GRAY, transform=ax.transAxes)

# ── Visual 1: Historical sales trend (row 1, col 0-1) ────────────────────
ax1 = fig.add_subplot(gs[1, 0:2])
monthly_hist = daily.set_index("date")["sales"].resample("MS").sum()
ax1.plot(monthly_hist.index, monthly_hist.values/1e6, color=BLUE, lw=2, marker="o", ms=3)
ax1.fill_between(monthly_hist.index, 0, monthly_hist.values/1e6, color=BLUE, alpha=0.08)
ax1.set_title("Historical Monthly Sales Trend", fontweight="bold", color=BLUE, fontsize=12)
ax1.set_ylabel("Sales (BDT millions)")
ax1.grid(axis="y", color="#eeeeee")
ax1.spines[['top','right']].set_visible(False)

# ── Visual 2: Forecasted sales (row 1, col 2-3) ───────────────────────────
ax2 = fig.add_subplot(gs[1, 2:4])
monthly['month_dt'] = pd.to_datetime(monthly['month_label'])
ax2.plot(monthly['month_dt'], monthly['baseline']/1e6, color=ORANGE, lw=2, marker="o", ms=4, label="Baseline forecast")
ax2.fill_between(monthly['month_dt'], monthly['conservative']/1e6, monthly['optimistic']/1e6,
                  color=ORANGE, alpha=0.15, label="Scenario range")
ax2.set_title("Forecasted Monthly Sales (Next 12M)", fontweight="bold", color=BLUE, fontsize=12)
ax2.set_ylabel("Sales (BDT millions)")
ax2.legend(fontsize=8)
ax2.grid(axis="y", color="#eeeeee")
ax2.spines[['top','right']].set_visible(False)
ax2.tick_params(axis='x', rotation=30)

# ── Visual 3: Monthly growth % (row 2, col 0-1) ───────────────────────────
ax3 = fig.add_subplot(gs[2, 0:2])
growth = monthly_hist.pct_change().dropna() * 100
colors_growth = [GREEN if g>=0 else RED for g in growth.values[-12:]]
ax3.bar(growth.index[-12:], growth.values[-12:], color=colors_growth, width=20)
ax3.axhline(0, color="grey", lw=0.8)
ax3.set_title("Month-over-Month Growth % (Last 12M)", fontweight="bold", color=BLUE, fontsize=12)
ax3.set_ylabel("Growth %")
ax3.grid(axis="y", color="#eeeeee")
ax3.spines[['top','right']].set_visible(False)
ax3.tick_params(axis='x', rotation=30)

# ── Visual 4: Scenario comparison (row 2, col 2-3) ────────────────────────
ax4 = fig.add_subplot(gs[2, 2:4])
scenario_totals = [monthly['conservative'].sum()/1e6, monthly['baseline'].sum()/1e6, monthly['optimistic'].sum()/1e6]
scenario_labels = ["Conservative\n(-10%)", "Baseline", "Optimistic\n(+10%)"]
bars = ax4.bar(scenario_labels, scenario_totals, color=[RED, BLUE, GREEN], alpha=0.85, width=0.55)
for bar, val in zip(bars, scenario_totals):
    ax4.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.3, f"{val:.1f}M", ha="center", fontsize=10, fontweight="bold")
ax4.set_title("12-Month Scenario Comparison", fontweight="bold", color=BLUE, fontsize=12)
ax4.set_ylabel("Total sales (BDT millions)")
ax4.grid(axis="y", color="#eeeeee")
ax4.spines[['top','right']].set_visible(False)

fig.text(0.02, 0.012, "Prepared by: Hasan Mahmud Sujan  |  Enterprise Sales Forecasting & Decision Support System",
          fontsize=8.5, color=GRAY)

fig.savefig(f"{DASH}/dashboard_mockup.png", dpi=180, bbox_inches="tight", facecolor="white")
plt.close(fig)
print("✔ Saved -> dashboard/dashboard_mockup.png")

# ════════════════════════════════════════════════════════════════════════
# MODEL CARD (governance documentation — matches JD requirement)
# ════════════════════════════════════════════════════════════════════════
model_card = {
    "model_name": "Enterprise Sales Forecasting Model",
    "version": "1.0.0",
    "date_trained": datetime.today().strftime("%Y-%m-%d"),
    "owner": "Hasan Mahmud Sujan",
    "objective": "Forecast daily/monthly aggregate sales to support inventory planning, procurement, and budget decisions",
    "data_period": f"{daily['date'].min().date()} to {daily['date'].max().date()}",
    "target_variable": "sales (daily aggregate, BDT)",
    "algorithm": "XGBoost Regressor (selected from 5 candidate models)",
    "candidate_models_evaluated": metrics["model"].tolist(),
    "validation_method": "Chronological train/test split (90-day hold-out)",
    "performance_holdout": {
        "MAE": float(metrics.iloc[0]["MAE"]),
        "RMSE": float(metrics.iloc[0]["RMSE"]),
        "MAPE_pct": float(metrics.iloc[0]["MAPE"]),
    },
    "features_used": [
        "lag_1, lag_7, lag_14, lag_28 (autoregressive lags)",
        "rolling_mean / rolling_std (7, 14, 30-day windows)",
        "calendar features (day_of_week, month, quarter, week_of_year)",
        "seasonal indicators (construction peak, post-monsoon peak, monsoon dip)",
        "promo_share, pct_stores_open (operational covariates)",
        "trend_days (linear time trend)",
    ],
    "scenario_methodology": {
        "baseline": "Direct model output, recursive 12-month forward forecast",
        "optimistic": "Baseline x 1.10",
        "conservative": "Baseline x 0.90",
    },
    "assumptions": [
        "Recent 90-day operational covariates (promo share, store openness) persist into the forecast horizon",
        "No major structural shocks (e.g. new competitor entry, regulatory change)",
        "Seasonal patterns observed in 2020-2025 continue to hold",
    ],
    "limitations": [
        "Recursive multi-step forecasting accumulates error over the horizon — accuracy degrades beyond ~90 days",
        "Does not explicitly model promotional calendar or marketing spend changes",
        "Scenario bands (±10%) are a simplified heuristic, not a statistically derived prediction interval",
        "ARIMA underperformed substantially (MAPE 28%) on this dataset — likely due to strong exogenous seasonality not captured by a univariate model",
    ],
    "retraining_trigger": "Rolling MAPE > 6% on most recent 30 days OR quarterly schedule, whichever comes first",
    "outputs": [
        "forecast_results.csv — test-period predictions vs actuals",
        "scenario_daily_forecast.csv / scenario_monthly_summary.csv — 12-month forward projections",
        "model_metrics.csv — full model comparison table",
        "dashboard_mockup.png — executive dashboard",
        "executive_summary.txt — management-facing narrative summary",
    ],
}

with open(f"{OUT}/model_card.json", "w") as f:
    json.dump(model_card, f, indent=2)
print("✔ Saved -> outputs/model_card.json")
