This repository implements a comprehensive Retail Analytics pipeline that spans the full lifecycle of data science in retail — from raw data to actionable insights and deployment.

It combines:

 1. Data preprocessing and feature engineering

 2. Exploratory Data Analysis (EDA)

 3. Store and department segmentation

 4. Market basket analysis

 5. Anomaly detection

 6. Sales forecasting (short- and long-term)

 7. Streamlit app for interactive forecasting and analysis


Retail Analytics/
├── .git/                          # Git version control metadata
├── .gitattributes
├── .gitignore
├── .venv/                         # Local virtual environment (optional)
│
├── data/
│   ├── raw/
│   │   ├── features.csv           # Store-level macroeconomic and metadata features
│   │   ├── sales.csv              # Weekly sales by store and department
│   │   └── stores.csv             # Store type, size, and location information
│   │
│   └── processed/
│       ├── cleaned_data.csv       # Cleaned dataset used for analysis
│       └── preprocessed_data.csv  # Final processed dataset ready for modeling
│
├── notebooks/
│   ├── EDA.ipynb
│   ├── Store Segmentation.ipynb
│   ├── Department-Segmentation.ipynb
│   ├── Market-Basket Analysis.ipynb
│   ├── Anomaly-Detection.ipynb
│   ├── Longterm-Demand-Forecasting.ipynb
│   ├── shortterm-Demand-Forecasting.ipynb
│   ├── global-model.ipynb
│   └── Error Analysis.ipynb
│
├── src/
│   ├── preprocess.py                     # Data preprocessing and feature creation
│   ├── global-forecasting-model.py       # Global forecasting model pipeline
│   ├── longterm-demand-forecasting.py    # Store-level long-term forecasting
│   ├── shortterm-demand-forecasting-copy.py # Short-term forecasting models
│   ├── dept_segmentation.py              # Department clustering logic
│   ├── store_segmentation.py             # Store-level segmentation
│   ├── marketbasket.py                   # Association rule mining pipeline
│   ├── anomaly_detection.py              # Time-series anomaly detection
│   └── utils.py                          # Common helper functions (if present)
│
├── results/
│   ├── global_model_metrics.csv          # RMSE, R², and parameters of global model
│   ├── longterm_best_models_summary.csv  # Best model per store summary
│   ├── longterm_forecast_results.csv     # Forecasts per store
│   ├── forecast_results.csv              # Combined forecasts summary
│   ├── shortterm_forecast_results_full.csv
│   ├── shortterm_forecast_results_tuned.csv
│   ├── top_rules.csv                     # Market basket rules (support/confidence/lift)
│   │
│   ├── Clustering using various methods.png
│   ├── Elbow method and silhouette score for number of clusters.png
│   ├── Departments with strong positive correlations.png
│   ├── Features for different clusters.png
│   ├── Network plot.png
│   ├── Support vs Confidence.png
│   ├── Lift between Antecedents and Consequents.png
│   ├── STK Decomposition along with rolling anomalies of Store 40, Dept 60.png
│   ├── chain_forecast_plot.png
│   ├── Longterm forecasting using LightGBM for store 10.png
│   ├── Longterm forecasting using XGBoost for store 10.png
│   ├── Longterm forecasting usingProphet for store 10.png
│   └── xgboost_predictions.png
│
├── saved_models/
│   ├── Chain/                          # Aggregated/global models
│   ├── Store/                          # Per-store models (LightGBM, Prophet, etc.)
│   └── short-term/Store/               # Short-term per-store models
│
├── forecasting-app.py                  # Streamlit dashboard for interactive forecasting
├── requirements.txt                    # Python dependencies
└── README.md                           # This documentation


**Problem Statement**

Retailers face three major challenges:

1. Demand forecasting: Predicting store-level weekly sales with high accuracy.

2. Behavioral segmentation: Understanding performance variation across stores and departments.

3. Cross-sell discovery: Identifying co-purchased product categories to improve promotions.

This project provides a unified analytics workflow to address all three using statistical and machine learning approaches.

**Data Flow & Methodology**
1. Data Preprocessing:

Merge sales, store, and feature datasets.

Handle missing values, outliers, and anomalies.

Create lag and rolling-window features for temporal modeling.

Generate holiday and event-based dummy variables.

2. Exploratory Data Analysis (EDA):

Analyze sales trends and seasonality by store and department.

Visualize macroeconomic effects (CPI, fuel price, unemployment).

Identify high-growth vs stagnant departments.

3. Segmentation:

Store segmentation via K-Means and hierarchical clustering.

Department segmentation via feature correlation and revenue patterns.

Used Silhouette and Elbow methods for optimal cluster selection.

![Store Clustering](results/Clustering%20using%20various%20methods.png)
![Cluster Features](results/Features%20for%20different%20clusters.png)

4. Market Basket Analysis

Implemented Apriori algorithm to find association rules.

Used metrics: support, confidence, and lift.

Focused on top 10 rules with high lift for actionable insights.
![Lift](results/Lift%20between%201-1%20Antecedents%20and%20Consequents.png)  
![Network Plot](results/Network%20plot.png)  
![Support vs Confidence](results/Support%20vs%20Confidence.png)  

5. Anomaly Detection

Decomposed sales series using STL.

Detected anomalies via rolling statistics + Isolation Forest.

Improves robustness of forecasting and model retraining.
[Isolation Forest and LOF]("results/results\anomaly_detection_IF_LOF.png") 
[STL Decomposition]("results/STK%20Decomposition%20along%20with%20rolling%20anomalies%20of%20Store%203,%20Dept%2025.png") 
  
6.Forecasting

Models: LightGBM, XGBoost, Prophet, and hybrid ensembles.

Metric	Value
RMSE	4779.57
R²	0.9526
Median RMSE (across stores)	~4800

- LightGBM and XGBoost outperform Prophet for long-term horizons.
- Longterm forecast
![XGBoost Forecast](results/Longterm%20forecasting%20using%20XGBoost%20for%20store%2010.png)  
![Prophet Forecast](results/Longterm%20forecasting%20usingProphet%20for%20store%2010.png)
-Shortterm Forecast
![SARIMAX Forecast](results/shortterm%20forecasting%20using%20sarimax%20for%20store%2010.png)
![LightGBM Forecast](results/shortterm%20forecasting%20using%20LightGBM%20for%20store%2010.png)  

7. Forecasting app
[App_screen_grab](results/app.pdf)  


**Run Instructions**
Step 1: Environment Setup
`python -m venv .venv`
`source .venv/bin/activate`  # Windows: `.venv\Scripts\activate`
`pip install -r requirements.txt`

Step 2: Preprocess Data
`python src/preprocess.py`

Step 3: Run Pipelines
`python src/longterm-demand-forecasting.py`
`python src/shortterm-demand-forecasting-copy.py`
`python src/marketbasket.py`
`python src/store_segmentation.py`
`python src/anomaly_detection.py`

Step 4: Launch Forecasting Dashboard
`streamlit run forecasting-app.py`
