"""
02_feature_engineering.py
Feature engineering for sales forecasting pipeline.

Author: Hasan Mahmud Sujan
"""
import pandas as pd
import numpy as np

print("=" * 65)
print("  FEATURE ENGINEERING")
print("=" * 65)

df = pd.read_csv("/home/claude/sales_forecast/data/raw_sales_data.csv", parse_dates=["date"])

# Aggregate to company-level daily sales (all stores combined) for the
# primary forecasting target — this is what management cares about most.
daily = (
    df.groupby("date")
      .agg(sales=("sales", "sum"),
           customers=("customers", "sum"),
           promo_share=("promo", "mean"),
           pct_stores_open=("open", "mean"))
      .reset_index()
)

# ── Calendar features ────────────────────────────────────────────────────
daily["year"]        = daily["date"].dt.year
daily["month"]       = daily["date"].dt.month
daily["quarter"]     = daily["date"].dt.quarter
daily["day_of_week"] = daily["date"].dt.dayofweek
daily["day_of_year"] = daily["date"].dt.dayofyear
daily["is_weekend"]  = daily["day_of_week"].isin([5, 6]).astype(int)
daily["week_of_year"]= daily["date"].dt.isocalendar().week.astype(int)

# ── Lag features ─────────────────────────────────────────────────────────
for lag in [1, 7, 14, 28]:
    daily[f"lag_{lag}"] = daily["sales"].shift(lag)

# ── Rolling statistics (using only past data — shift(1) avoids leakage) ──
for window in [7, 14, 30]:
    daily[f"rolling_mean_{window}"] = daily["sales"].shift(1).rolling(window).mean()
    daily[f"rolling_std_{window}"]  = daily["sales"].shift(1).rolling(window).std()

# ── Seasonal indicators ──────────────────────────────────────────────────
daily["is_construction_peak"] = daily["month"].isin([3,4,5,6]).astype(int)
daily["is_post_monsoon_peak"] = daily["month"].isin([10,11]).astype(int)
daily["is_monsoon_dip"]       = daily["month"].isin([7,8]).astype(int)

# ── Trend feature (days since start) ─────────────────────────────────────
daily["trend_days"] = (daily["date"] - daily["date"].min()).dt.days

# Drop initial rows with NaN lag/rolling features
daily_clean = daily.dropna().reset_index(drop=True)

daily.to_csv("/home/claude/sales_forecast/data/processed_sales_data.csv", index=False)

print(f"\nDaily aggregated dataset : {daily.shape[0]:,} rows x {daily.shape[1]} columns")
print(f"After dropping NaN warmup: {daily_clean.shape[0]:,} rows")
print(f"\nFeatures created:")
feat_cols = [c for c in daily.columns if c not in ["date","sales","customers"]]
for c in feat_cols:
    print(f"  - {c}")

print("\n✔ Saved -> data/processed_sales_data.csv")
