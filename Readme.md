# 🛒 Retail Analytics — End-to-End Retail Intelligence Platform

**Generated:** Important project README to describe the repo, structure, methods, results and how to reproduce results.

---

## 📘 Overview
This repository implements a full retail analytics pipeline from raw sales data to business-ready outputs:

- Exploratory Data Analysis (EDA) at store & department granularity  
- Store & department **segmentation** (clustering)  
- **Market-basket / association rule** mining for cross-sell insight  
- **Anomaly detection** (STL/STK decomposition + Isolation Forest / LOF)  
- Short-term and long-term **forecasting** (XGBoost, LightGBM, Prophet, hybrid)  
- An interactive **Streamlit forecasting dashboard** (`forecasting-app.py`) for stakeholders

---


---

##  Problem statement
Retail chains need accurate, interpretable demand forecasts at the store × department level, plus behavioral insights to guide promotions, inventory and placement decisions. This repo addresses:

- Forecast weekly sales (short and long horizons) for stores & departments  
- Group stores/departments into actionable segments (for targeted strategies)  
- Discover product/department co-purchases (market-basket rules) to increase basket size  
- Detect anomalies in sales time-series so forecasts are robust and business actions can be taken quickly

---

##  Data & Core Inputs
- `data/raw/sales.csv` — master timeseries of weekly sales by Store and Department  
- `data/raw/stores.csv` — store metadata (type, location, size, etc.)  
- `data/raw/features.csv` — merged features including macro indicators (CPI, fuel price, unemployment)  
- `data/processed/` — cleaned and preprocessed datasets used by models and notebooks

---

##  Methodology / Pipeline
1. **Preprocessing** (`src/preprocess.py`)  
   - Join sales with store metadata and macro indicators  
   - Impute missing values, fix outliers
2. **EDA** (`notebooks/EDA.ipynb`)  
   - Inspect seasonality, store-wise trends, department contributions, correlations with macros  
3. **Segmentation** (`src/store_segmentation.py`, `notenooks/Dept-Segmentation.ipynb`)  
   - KMeans; Elbow & Silhouette analysis; visualize using PCA  
4. **Market-basket Analysis** (`src/marketbasket.py`)  
   - Aggregate transactional data to department-level baskets; run Apriori / association-rule mining; compute support/confidence/lift  
5. **Anomaly Detection** (`src/anomaly_detection.py`)  
   - Time-series decomposition (STL/STK), rolling z-score, Isolation Forest / LOF 
6. **Forecasting**  
   - **Global model**: single model trained across all series (global-forecasting-model.py)  
   - **Per-store** long-term: XGBoost  / Prophet pipelines (longterm-demand-forecasting.py)  
   - **Short-term** SARIMAX / LightGBM (shortterm-demand-forecasting-copy.py)  
   - Model evaluation with RMSE / MAE / R²; per-store best model recorded in `results/longterm_best_models_summary.csv`  
7. **Deployment**: `forecasting-app.py` (Streamlit) loads `results/*` files and saved models to provide interactive forecasts

---

## Results — concise summary (from `results/`)
> These numbers / file references are taken from CSVs in `results/` included in the repository.

**Global model (results/global_model_metrics.csv)**  
| Metric | Value |
|--------|-------:|
| RMSE   | 4,779.57 |
| R²     | 0.9526 |

**Per-store performance**  
- **Median RMSE across stores:** ≈ 4,800 (see `results/longterm_best_models_summary.csv`)  
- Per-store best-models and their chosen hyperparameters are in `results/longterm_best_models_summary.csv` and `results/longterm_forecast_results.csv`.

**Forecast files (examples)**  
- `results/forecast_results.csv` — consolidated forecast outputs  
- `results/longterm_forecast_results.csv` — long-horizon forecasts per store  
- `results/shortterm_forecast_results_full.csv` / `shortterm_forecast_results_tuned.csv` — short-term forecasts & tuned variants
Long term forecast results(example : Store 10)
- `results/Longterm forecasting using LightGBM for store 10.png`
- `results/Longterm forecasting using XGBoost for store 10.png`
Short term forecast results(example : Store 10):
-`results/shortterm forecasting using sarimax for store 10.png`
-`results/shortterm forecasting using LightGBM for store 10.png`

---

##  Segmentation (Store & Department)
**What was done**
- KMeans and hierarchical clustering were run using features such as mean weekly sales, volatility (std), seasonality strength, and macro sensitivity.
- Elbow and silhouette analysis used to choose cluster count.

**Key clusters (illustrative)**  
- **Cluster 0 — Flagship / High-volume stores:** Big-volume stores/depts with seasonal peaks, volatile demand, and more anomalies. 

- **Cluster 1 — Low-volume / Seasonal stores:**Smaller, stable stores/depts with predictable sales. Markdown-driven — promotions play a stronger role in boosting performance.

**Files & visualizations**
- `results/Clustering using various methods.png`  
- `results/Elbow method and silhouette score for number of clusters.png`  
- `results/Features for different clusters.png`

**Business use**
- Different forecasting strategies or inventory buffers by cluster; different promotional policies.

---

##  Market-basket Analysis
**What was done**
- Association rule mining on transactional data aggregated to department-level baskets.
- Rules filtered by support/confidence/lift thresholds to extract actionable co-purchases.

**Top rules** (sample / illustrative — full list in `results/top_rules.csv`):


**Visuals**
- `results/Network plot.png`  
- `results/Support vs Confidence.png`  
- `results/Lift between 1-1 Antecedents and Consequents.png`
- `results/Network plot.png`

**Business use**
- Use for cross-promotion, bundle suggestions, and store layout decisions.

---

##  Anomaly Detection
**Process**
- STL/STK decomposition per store-department, rolling anomaly detection on residuals, then flagged via Isolation Forest / LOF.

**Example**
- `results/STK Decomposition along with rolling anomalies of Store 40, Dept 60.png` — shows flagged weeks.

**Impact**
- Removes/flags outlier weeks before training to improve model stability and to trigger business investigations (e.g., incorrect promo label, data error, or real surge).

---

##  Forecasting (Details & Visuals)
**Model families**
- Gradient-boosted tree regressors: LightGBM & XGBoost (feature-rich models)  
- Time series univariate: Prophet (per-store)  
- Hybrid: ML model for baseline + time-series residual models

**Representative visuals**
- `results/shortterm forecasting using LightGBM for store 10.png`  
- `results/Longterm forecasting using XGBoost for store 10.png`  
- `results/Longterm forecasting usingProphet for store 10.png`  
- `results/shortterm forecasting using sarimax for store 10.png`
- `results/xgboost_predictions.png` (pred vs actual / residuals)  
- 

**Evaluation**
- Compare models on RMSE / MAE / R²; pick best per-store and save model artifacts in `saved_models/Store/` and `saved_models/short-term/Store/`.

**Deployment**
- `forecasting-app.py` (Streamlit) loads `results/*` files and saved models to provide interactive forecasts.
-Example forecasting app screenshot :`results/app.pdf`

---

##  Reproducibility — how to run everything
> Note: if your repo root has a space in the folder name, either quote paths or rename folder to `Retail_Analytics`.

1. Create a virtual environment and install:
```bash
python -m venv .venv
# Linux / macOS:
source .venv/bin/activate
# Windows (PowerShell):
# .venv\Scripts\Activate.ps1

pip install -r requirements.txt


