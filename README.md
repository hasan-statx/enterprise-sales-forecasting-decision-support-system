# Enterprise Sales Forecasting & Decision Support System

**Author:** Hasan Mahmud Sujan  
**Stack:** Python · XGBoost · Prophet · ARIMA (statsmodels) · scikit-learn · matplotlib

---

## Executive Overview

Accurate demand forecasting is critical for inventory planning, procurement optimisation, budget allocation, and strategic business decision-making.

This project develops a business forecasting and decision-support framework capable of predicting future sales patterns while generating actionable management insights through scenario analysis and performance monitoring — built around a multi-store retail/FMCG sales dataset with realistic seasonality (construction-material demand cycle, Eid periods, monsoon dip).

---

## Business Questions

1. What are expected sales for the next 3, 6, and 12 months?
2. Which operational and calendar factors influence sales performance?
3. What risks exist under optimistic and conservative business scenarios?
4. How can management improve planning accuracy and resource allocation?

---

## Dataset

Synthetic multi-store daily sales data (20 stores, 2020–2025) engineered to reflect realistic retail dynamics:

- Long-term trend (~6% YoY growth)
- Weekly seasonality (Friday dip, weekend peak)
- Yearly seasonality (construction-material demand: Mar–Jun & Oct–Nov peaks; monsoon dip: Jul–Aug)
- Bangladesh public holiday and Eid-period effects
- Random promotions (~12% of days)
- Store-type heterogeneity (4 store types, 3 assortment levels)

| File | Description |
|------|-------------|
| `data/raw_sales_data.csv` | Daily sales by store (43,840 rows) |
| `data/store_metadata.csv` | Store-level attributes |
| `data/processed_sales_data.csv` | Company-level daily aggregate with engineered features |

---

## Methodological Framework

### 1. Data Preparation
- Aggregation to company-level daily sales
- Missing value handling via warmup-period exclusion
- Outlier behaviour driven by genuine calendar effects (holidays, Eid) rather than removed

### 2. Feature Engineering
- **Lag features:** 1, 7, 14, 28 days
- **Rolling statistics:** 7/14/30-day rolling mean & std (computed on past data only — no leakage)
- **Calendar features:** day-of-week, month, quarter, week-of-year
- **Seasonal indicators:** construction peak, post-monsoon peak, monsoon dip
- **Trend feature:** linear day-count since series start

### 3. Forecasting Models (5 candidates evaluated)
| Model | Type |
|-------|------|
| Linear Regression | Feature-based baseline |
| Random Forest | Ensemble tree-based |
| **XGBoost** | Gradient-boosted trees — **best performer** |
| ARIMA(5,1,2) | Classical univariate time-series |
| Prophet | Additive decomposition with regressors |

### 4. Validation
Chronological train/test split (90-day hold-out — no random shuffling, avoiding look-ahead bias). Metrics: **MAE, RMSE, MAPE**.

### 5. Scenario Analysis
12-month recursive forward forecast under three business scenarios:
- **Baseline** — direct model output
- **Optimistic** (+10%)
- **Conservative** (−10%)

### 6. Executive Reporting
- Power BI-style dashboard mockup (KPI cards + 4 visuals)
- Model card (`model_card.json`) documenting objective, features, assumptions, limitations, and retraining triggers
- Plain-language executive summary for non-technical stakeholders

---

## Results

| Model | MAE | RMSE | MAPE |
|-------|-----|------|------|
| **XGBoost** | **4,885** | **6,045** | **3.30%** |
| Prophet | 3,926 | 5,405 | 4.30% |
| Linear Regression | 5,710 | 7,497 | 4.60% |
| Random Forest | 6,611 | 8,868 | 4.71% |
| ARIMA(5,1,2) | 21,378 | 30,165 | 28.18% |

**XGBoost selected as the production model** — lowest MAPE on the 90-day hold-out test.

### 12-Month Forward Scenario Totals
| Scenario | Total Sales (BDT) |
|----------|--------------------|
| Conservative (−10%) | 17,312,210 |
| **Baseline** | **19,235,790** |
| Optimistic (+10%) | 21,159,368 |

**Key seasonal finding:** peak demand expected in March; lowest demand expected in July (monsoon dip), consistent with the construction-material demand cycle embedded in the data.

---

## Repository Structure

```
sales_forecast/
│
├── data/
│   ├── raw_sales_data.csv
│   ├── store_metadata.csv
│   └── processed_sales_data.csv
│
├── src/
│   ├── 01_generate_data.py
│   ├── 02_feature_engineering.py
│   ├── 03_forecasting_models.py
│   ├── 04_scenario_analysis.py
│   └── 05_executive_dashboard.py
│
├── outputs/
│   ├── model_metrics.csv
│   ├── forecast_results.csv
│   ├── forecast_plot.png
│   ├── model_comparison.png
│   ├── scenario_daily_forecast.csv
│   ├── scenario_monthly_summary.csv
│   ├── scenario_analysis.png
│   ├── executive_summary.txt
│   └── model_card.json
│
├── dashboard/
│   └── dashboard_mockup.png
│
├── README.md
└── requirements.txt
```

---

## How to Run

```bash
pip install -r requirements.txt

python src/01_generate_data.py
python src/02_feature_engineering.py
python src/03_forecasting_models.py
python src/04_scenario_analysis.py
python src/05_executive_dashboard.py
```

---

## Model Documentation (Governance)

A full `model_card.json` is generated, covering:
- Objective, data period, target variable
- Algorithm and candidate models evaluated
- Hold-out performance (MAE / RMSE / MAPE)
- Features used
- Scenario methodology
- Assumptions and limitations
- Retraining trigger criteria

This follows a model-governance structure suitable for enterprise BI/analytics teams — documenting not just *what* the model predicts, but *when it should be retrained* and *where it is likely to fail*.

---

## Skills Demonstrated

- Time-series feature engineering without data leakage
- Multi-model benchmarking (classical statistical vs. ML vs. additive decomposition)
- Chronological validation methodology
- Scenario-based decision support for business planning
- Executive dashboard design and non-technical communication
- Model documentation / governance practices
