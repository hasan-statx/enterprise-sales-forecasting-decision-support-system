"""
01_generate_data.py
Synthetic Multi-Store Retail Sales Dataset (Rossmann-style)
=============================================================
Simulates daily sales across multiple stores with realistic
trend, weekly seasonality, yearly seasonality, promotions,
holidays, and store-level heterogeneity — designed to mirror
the structure of the Kaggle "Rossmann Store Sales" dataset.

Author: Hasan Mahmud Sujan
"""
import numpy as np
import pandas as pd

np.random.seed(42)

N_STORES = 20
START = "2020-01-01"
END   = "2025-12-31"
STORE_TYPES = ["a", "b", "c", "d"]
ASSORTMENT  = ["basic", "extra", "extended"]

dates = pd.date_range(START, END, freq="D")

# ── Store metadata ───────────────────────────────────────────────────────
stores = pd.DataFrame({
    "store_id": range(1, N_STORES + 1),
    "store_type": np.random.choice(STORE_TYPES, N_STORES, p=[.4, .2, .25, .15]),
    "assortment": np.random.choice(ASSORTMENT, N_STORES, p=[.5, .15, .35]),
    "competition_distance_m": np.random.exponential(2500, N_STORES).round(0) + 100,
    "base_daily_sales": np.random.normal(5500, 1200, N_STORES).round(0).clip(2000, None),
})

# ── Bangladesh-relevant holiday calendar (for an FMCG/building-materials business) ──
def get_holidays(years):
    holidays = []
    for y in years:
        holidays += [
            f"{y}-01-01",  # New Year
            f"{y}-02-21",  # Language Martyrs Day
            f"{y}-03-26",  # Independence Day
            f"{y}-04-14",  # Pohela Boishakh
            f"{y}-05-01",  # May Day
            f"{y}-08-15",  # National Mourning Day
            f"{y}-12-16",  # Victory Day
            f"{y}-12-25",  # Christmas
        ]
    return pd.to_datetime(holidays)

years = range(2020, 2026)
holiday_dates = set(get_holidays(years))

# Eid dates (approximate, lunar — shifts ~11 days earlier each year)
eid_ul_fitr = pd.to_datetime(["2020-05-24","2021-05-13","2022-05-02",
                               "2023-04-21","2024-04-10","2025-03-30"])
eid_ul_adha = pd.to_datetime(["2020-07-31","2021-07-20","2022-07-09",
                               "2023-06-28","2024-06-16","2025-06-06"])

records = []
for _, store in stores.iterrows():
    sid = store["store_id"]
    base = store["base_daily_sales"]

    # Construction-material demand peaks: Mar-Jun (dry season), Oct-Nov (post-monsoon)
    for d in dates:
        month = d.month
        dow = d.dayofweek  # 0=Mon

        # Trend: ~6% YoY growth + store-level drift
        years_elapsed = (d - pd.Timestamp(START)).days / 365.25
        trend = 1 + 0.06 * years_elapsed

        # Yearly seasonality (construction-driven)
        if month in [3, 4, 5, 6]:
            season = 1.22
        elif month in [10, 11]:
            season = 1.12
        elif month in [7, 8]:           # monsoon dip
            season = 0.72
        else:
            season = 1.0

        # Weekly seasonality — Friday lower (weekend prep), Sat/Sun higher footfall
        dow_factor = {0: 1.05, 1: 1.02, 2: 1.0, 3: 1.0,
                      4: 0.78, 5: 1.18, 6: 1.10}[dow]

        # Holiday effect
        is_public_holiday = d in holiday_dates
        is_eid = (d in eid_ul_fitr) or (d in eid_ul_adha)
        days_to_eid = min([abs((d - e).days) for e in list(eid_ul_fitr) + list(eid_ul_adha)])
        eid_buildup = 1.35 if (days_to_eid <= 3 and days_to_eid > 0) else 1.0
        holiday_factor = 0.35 if (is_public_holiday or is_eid) else eid_buildup

        # Random promotions (~12% of days)
        promo = np.random.random() < 0.12
        promo_factor = 1.28 if promo else 1.0

        # Store-type modifier
        type_factor = {"a": 1.0, "b": 0.85, "c": 1.15, "d": 0.95}[store["store_type"]]

        noise = np.random.normal(1.0, 0.09)

        sales = max(0, base * trend * season * dow_factor *
                    holiday_factor * promo_factor * type_factor * noise)

        # Occasionally store closed (rare random closures + all major holidays)
        closed = is_public_holiday and np.random.random() < 0.6
        if closed:
            sales = 0

        customers = max(0, int(sales / np.random.uniform(280, 420)))

        records.append({
            "date": d,
            "store_id": sid,
            "store_type": store["store_type"],
            "assortment": store["assortment"],
            "sales": round(sales, 2),
            "customers": customers,
            "promo": int(promo),
            "school_holiday": int(is_public_holiday),
            "eid_period": int(is_eid or (days_to_eid <= 3)),
            "day_of_week": dow,
            "open": int(not closed),
        })

df = pd.DataFrame(records)
df = df.merge(stores[["store_id","competition_distance_m"]], on="store_id")

df.to_csv("/home/claude/sales_forecast/data/raw_sales_data.csv", index=False)
stores.to_csv("/home/claude/sales_forecast/data/store_metadata.csv", index=False)

print(f"Generated {len(df):,} rows across {N_STORES} stores")
print(f"Date range: {df['date'].min().date()} to {df['date'].max().date()}")
print(f"Total sales (all stores, all time): {df['sales'].sum():,.0f} BDT")
print(df.groupby('store_id')['sales'].mean().describe())
