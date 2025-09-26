import pandas as pd
import numpy as np
from pathlib import Path
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_squared_error, r2_score
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
from prophet import Prophet
import matplotlib.pyplot as plt
import warnings
from statsmodels.tools.sm_exceptions import ConvergenceWarning
from tqdm import tqdm
import optuna
import json


warnings.simplefilter("ignore", ConvergenceWarning)
warnings.simplefilter("ignore", UserWarning)

RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)

# -------------------------------
# Log vs Raw evaluation helper
# -------------------------------
def evaluate_log_vs_raw(y, model_type="sarima", horizon=8, order=(1,1,1), seasonal_order=(1,1,1,52)):
    """Evaluate ARIMA/SARIMA with raw vs log-transformed target"""
    results = []

    train_size = int(len(y)*0.8)
    y_train, y_test = y.iloc[:train_size], y.iloc[train_size:]

    for transform in ["raw", "log"]:
        if transform == "raw":
            y_train_fit = y_train
        else:
            y_train_fit = np.log1p(y_train)

        seasonal_order_use = seasonal_order if model_type=="sarima" else (0,0,0,0)
        model = SARIMAX(y_train_fit, order=order, seasonal_order=seasonal_order_use,
                        enforce_stationarity=False, enforce_invertibility=False)
        fitted = model.fit(disp=False)

        preds = fitted.forecast(steps=len(y_test))
        if transform == "log":
            preds = np.expm1(preds)

        rmse = np.sqrt(mean_squared_error(y_test, preds))
        r2 = r2_score(y_test, preds)

        results.append({"Transform": transform, "RMSE": rmse, "R2": r2})

    # Pick best by RMSE
    best = min(results, key=lambda x: x["RMSE"])
    return best

# -------------------------------
# ARIMA + SARIMA for Store & Dept
# -------------------------------
def arima_sarima_store_and_dept(df, horizon=8):
    results = []

    df = df[['Store', 'Dept', 'Date', 'Weekly_Sales']].copy()

    # Store-level
    for store in tqdm(df['Store'].unique()):
        series = (df[df['Store']==store].groupby('Date')['Weekly_Sales']
                  .sum().asfreq('W').fillna(0).sort_index())
        if len(series) < horizon*3:
            continue
        # ARIMA
        best_arima = evaluate_log_vs_raw(series, model_type="arima", horizon=horizon)
        results.append({"Level":"Store","ID":store,"Horizon":"Short-term",
                        "Model":"ARIMA","RMSE":best_arima["RMSE"],"R2":best_arima["R2"],
                        "Transform":best_arima["Transform"]})
        # SARIMA
        best_sarima = evaluate_log_vs_raw(series, model_type="sarima", horizon=horizon)
        results.append({"Level":"Store","ID":store,"Horizon":"Short-term",
                        "Model":"SARIMA","RMSE":best_sarima["RMSE"],"R2":best_sarima["R2"],
                        "Transform":best_sarima["Transform"]})

    # # Dept-level
    # print("Depat level aggregation......")
    # for dept in tqdm(df['Dept'].unique()):
    #     series = (df[df['Dept']==dept].groupby('Date')['Weekly_Sales']
    #               .sum().asfreq('W').fillna(0).sort_index())
    #     if len(series) < horizon*3:
    #         continue
    #     # ARIMA
    #     best_arima = evaluate_log_vs_raw(series, model_type="arima", horizon=horizon)
    #     results.append({"Level":"Dept","ID":dept,"Horizon":"Short-term",
    #                     "Model":"ARIMA","RMSE":best_arima["RMSE"],"R2":best_arima["R2"],
    #                     "Transform":best_arima["Transform"]})
    #     # SARIMA
    #     best_sarima = evaluate_log_vs_raw(series, model_type="sarima", horizon=horizon)
    #     results.append({"Level":"Dept","ID":dept,"Horizon":"Short-term",
    #                     "Model":"SARIMA","RMSE":best_sarima["RMSE"],"R2":best_sarima["R2"],
    #                     "Transform":best_sarima["Transform"]})

    return pd.DataFrame(results)

# -------------------------------
# LightGBM for Store–Dept
# -------------------------------
def lightgbm_forecast(df, store, dept, horizon=8):

    df = df[['Store','Dept','Date','Weekly_Sales','IsHoliday','Temperature','Fuel_Price', \
            'cpi','unemployment','Size','Year','Month','WeekOfYear', \
            'Total_MarkDown','store_mean','sales_vs_store',\
            'sales_vs_dept']]
    
    group = df[(df['Store']==store)&(df['Dept']==dept)].copy()

    for lag in [1,2,4,8,12,52]:  # include yearly lag for long-term
        group[f'lag_{lag}'] = group['Weekly_Sales'].shift(lag)

    for window in [4,12,52]:
        group[f'roll_mean_{window}'] =group['Weekly_Sales'].shift(1).rolling(window).mean()

    
    group = group.dropna().sort_values("Date")

    if len(group) < horizon*3:
        return None
    X = group.drop(columns=["Weekly_Sales","Date","Store","Dept"])
    y = group["Weekly_Sales"]
    train_size = int(len(X)*0.8)
    X_train, X_test = X.iloc[:train_size], X.iloc[train_size:]
    y_train, y_test = y.iloc[:train_size], y.iloc[train_size:]
    model = LGBMRegressor(
    n_estimators=500,        # more boosting rounds
    learning_rate=0.05,      # small steps for stability
    num_leaves=31,           # controls tree complexity
    max_depth=-1,            # let trees grow automatically
    min_data_in_leaf=10,     # allow small leaves (captures short-term changes)
    feature_fraction=0.9,    # random feature sampling for regularization
    bagging_fraction=0.8,    # random row sampling for regularization
    bagging_freq=5,          # perform bagging every 5 iterations
    lambda_l1=1.0,           # L1 regularization
    lambda_l2=1.0,           # L2 regularization
    random_state=42)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    return {"Level":"Store-Dept","Store":store,"Dept":dept,
            "Horizon":"Short-term","Model":"LightGBM",
            "RMSE":np.sqrt(mean_squared_error(y_test, preds)),
            "R2":r2_score(y_test, preds),
            "Transform":"raw"}  # ML models don’t need log scaling


# -------------------------------
# Long-term: Prophet
# -------------------------------
def prophet_forecast(df, horizon=52, level="Chain", store=None):

    df = df[['Store','Dept','Date','Weekly_Sales','IsHoliday',\
        'cpi','unemployment', 'Fuel_Price', \
        'Total_MarkDown']].copy()
    
    if level=="Chain":
            agg = df.groupby('Date').agg({
            'Weekly_Sales':'sum',
            'cpi':'mean',
            'unemployment':'mean',
            'Fuel_Price':'mean',
            'Total_MarkDown':'sum',
            'IsHoliday':'mean'
        }).reset_index()
    elif level=="Store":
          
        agg = df[df['Store']==store].groupby('Date').agg({
            'Weekly_Sales':'sum',
            'cpi':'mean',
            'unemployment':'mean',
            'Fuel_Price':'mean',
            'Total_MarkDown':'sum',
            'IsHoliday':'mean'
        }).reset_index()
    else: return None
    agg = agg.rename(columns={"Date":"ds","Weekly_Sales":"y"})
    
    # Train/test split for evaluation
    train_size = int(len(agg)*0.8)
    train, test = agg.iloc[:train_size], agg.iloc[train_size:]
    m = Prophet(weekly_seasonality=True, yearly_seasonality=True)
    for reg in ["cpi","unemployment","Fuel_Price", "Total_MarkDown", "IsHoliday"]:
        if reg in df.columns:
            m.add_regressor(reg)

        
    m.fit(train)
    future = m.make_future_dataframe(periods=horizon, freq="W")
    for reg in ["cpi","unemployment","Fuel_Price","Total_MarkDown","IsHoliday"]:
        if reg in agg.columns:
            future.loc[:train_size-1, reg] = train[reg].values
            # Forecast horizon part → repeat last known value
            future.loc[train_size:, reg] = train[reg].iloc[-1]
    forecast = m.predict(future)

    preds = forecast.iloc[-len(test):]["yhat"].values
    y_true = test["y"].values


    rmse = np.sqrt(mean_squared_error(y_true, preds))
    r2 = r2_score(y_true, preds)

    return {
        "Level": level,
        "Store": store,
        "Horizon": "Long-term",
        "Model": "Prophet",
        "RMSE": rmse,
        "R2": r2,
        "Transform": "raw",
        "Forecast": forecast   # optional: keep full forecast for plotting
    }

# -------------------------------
# Long-term: XGBoost (Recursive)
# -------------------------------


dRESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)


# -------------------------
# Walk-forward validation
# -------------------------
def walk_forward_validation(X, y, params, use_log=True, horizon=26):
    """
    Expanding window walk-forward validation for long-term forecasting.

    horizon = number of weeks predicted in each step (default=26 weeks ~ 6 months).
    """
    n_total = len(y)
    window_size = int(n_total * 0.7)  # start with 70% training
    rmse_scores = []

    for start in range(window_size, n_total - horizon, horizon):
        X_train, y_train = X.iloc[:start], y.iloc[:start]
        X_val, y_val = X.iloc[start:start+horizon], y.iloc[start:start+horizon]

        # store_mean = y_train.mean()
        # X_train = X_train.copy()
        # X_val = X_val.copy()

        # Ratios
        # X_train.loc[:, "sales_vs_store"] = y_train / store_mean
        # X_val.loc[:, "sales_vs_store"] = y_val / store_mean

        y_train_model = np.log1p(y_train) if use_log else y_train

        model = XGBRegressor(**params)
        model.fit(
            X_train, y_train_model,
            eval_set=[(X_val, np.log1p(y_val) if use_log else y_val)],
            # early_stopping_rounds=50,
            verbose=False
        )

        y_pred = model.predict(X_val)
        if use_log:
            y_pred = np.expm1(y_pred)

        rmse = np.sqrt(mean_squared_error(y_val, y_pred))
        rmse_scores.append(rmse)

    return np.mean(rmse_scores)


# -------------------------
# Run tuning per store
# -------------------------
def tune_xgb_per_store(df, horizon=26, n_trials=30):

    df = df[['Store','Dept','Date','Weekly_Sales','IsHoliday',\
          'sales_vs_store',
    'sales_vs_dept', \
        'Total_MarkDown']].copy()
    num_of_stores = df['Store'].nunique()
    results = []

    for i in tqdm(range(1, num_of_stores + 1)):
        print(f"\n=== Store {i} ===")

        # --- Aggregate sales ---
        group = df[df['Store'] == i].copy()
        group["date"] = pd.to_datetime(group["Date"])
       
        # --- Aggregate sales + exogenous features ---
        group = group.groupby("date").agg({
            "Weekly_Sales": "sum"
            # "IsHoliday": "mean",        
            # "Total_MarkDown": "sum"
          
        }).reset_index()
        
        group = group.sort_values("date").set_index("date")

        # --- Lag features ---
        group["lag_1"] = group["Weekly_Sales"].shift(1)
        group["lag_2"] = group["Weekly_Sales"].shift(2)
        group["lag_52"] = group["Weekly_Sales"].shift(52)

        # --- Rolling features ---
        for window in [4, 12, 52]:
            group[f"roll_mean_{window}"] = group["Weekly_Sales"].shift(1).rolling(window).mean()

        # --- Calendar features ---
        group["weekofyear"] = group.index.isocalendar().week.astype(int)
        group["month"] = group.index.month
        group["year"] = group.index.year

        group = group.dropna()
        X = group.drop(columns=["Weekly_Sales"])
        y = group["Weekly_Sales"]

        # --- Optuna objective ---
        def objective(trial):
            params = {
                "objective": "reg:squarederror",
                "n_estimators": trial.suggest_int("n_estimators", 200, 800),
                "max_depth": trial.suggest_int("max_depth", 3, 10),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
                "subsample": trial.suggest_float("subsample", 0.6, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
                "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
                "gamma": trial.suggest_float("gamma", 0.0, 2.0),
                "reg_lambda": trial.suggest_float("reg_lambda", 0.0, 5.0),
                "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 5.0),
                "random_state": 42
            }

            use_log = trial.suggest_categorical("use_log", [True, False])
            return walk_forward_validation(X, y, params, use_log=use_log, horizon=horizon)

        # --- Run Optuna study ---
        study = optuna.create_study(direction="minimize")
        study.optimize(objective, n_trials=n_trials, n_jobs=-1)

        best_params = study.best_params
        use_log = best_params.pop("use_log")
        print("Best params:", best_params, "Log transform:", use_log)

        # --- Final evaluation on last 20% ---
        train_size = int(len(X) * 0.8)
        X_train, X_test = X.iloc[:train_size], X.iloc[train_size:]
        y_train, y_test = y.iloc[:train_size], y.iloc[train_size:]

        y_train_final = np.log1p(y_train) if use_log else y_train

        best_model = XGBRegressor(**best_params)
        best_model.fit(X_train, y_train_final)

        y_pred = best_model.predict(X_test)
        if use_log:
            y_pred = np.expm1(y_pred)

        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)

        print(f"Final tuned model → RMSE: {rmse:.2f}, R²: {r2:.3f}")

        results.append({
            "Store": i,
            "RMSE": rmse,
            "R2": r2,
            "Best_Params": best_params,
            "Transform": "log" if use_log else "raw"
        })

        # Save best params
        with open(RESULTS_DIR / 'XGBResults'/f"store_{i}_best_params.json", "w") as f:
            json.dump({"params": best_params, "use_log": use_log}, f)

    return results

def tune_xgb_chain(df, horizon=26, n_trials=30):
    """
    Tune and evaluate XGBoost for chain-level forecasting using Optuna + walk-forward validation.
    """

    
    df = df[['Store','Dept','Date','Weekly_Sales','IsHoliday',\
        'sales_vs_store',\
            'sales_vs_dept', \
        'Total_MarkDown']].copy()

    # --- Aggregate sales at chain level ---
    group = df.copy()
    group["date"] = pd.to_datetime(group["Date"])
    group = group.groupby("date").agg({
    "Weekly_Sales": "sum"
    # "IsHoliday": "mean",        
    # "Total_MarkDown": "sum",
}).reset_index()
    group = group.sort_values("date").set_index("date")

    # Lag features
    group["lag_1"] = group["Weekly_Sales"].shift(1)
    group["lag_2"] = group["Weekly_Sales"].shift(2)
    group["lag_52"] = group["Weekly_Sales"].shift(52)

    # Rolling features
    for window in [4, 12, 52]:
        group[f"roll_mean_{window}"] = group["Weekly_Sales"].shift(1).rolling(window).mean()

    # Calendar features
    group["weekofyear"] = group.index.isocalendar().week.astype(int)
    group["month"] = group.index.month
    group["year"] = group.index.year

    group = group.dropna()
    X = group.drop(columns=["Weekly_Sales"])
    y = group["Weekly_Sales"]

    # --- Optuna objective ---
    def objective(trial):
        params = {
            "objective": "reg:squarederror",
            "n_estimators": trial.suggest_int("n_estimators", 200, 800),
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
            "gamma": trial.suggest_float("gamma", 0.0, 2.0),
            "reg_lambda": trial.suggest_float("reg_lambda", 0.0, 5.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 5.0),
            "random_state": 42
        }
        use_log = trial.suggest_categorical("use_log", [True, False])
        return walk_forward_validation(X, y, params, use_log=use_log, horizon=horizon)

    # --- Run study ---
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=n_trials, n_jobs=-1)

    best_params = study.best_params
    use_log = best_params.pop("use_log")

    print("Best params for Chain:", best_params, "Log transform:", use_log)

    # --- Final evaluation on last 20% ---
    train_size = int(len(X) * 0.8)
    X_train, X_test = X.iloc[:train_size], X.iloc[train_size:]
    y_train, y_test = y.iloc[:train_size], y.iloc[train_size:]

    y_train_final = np.log1p(y_train) if use_log else y_train

    best_model = XGBRegressor(**best_params)
    best_model.fit(X_train, y_train_final)

    y_pred = best_model.predict(X_test)
    if use_log:
        y_pred = np.expm1(y_pred)

    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    return {
        "Level": "Chain",
        "Horizon": f"{horizon} weeks",
        "Model": f"XGBoost_{'log' if use_log else 'raw'}",
        "RMSE": rmse,
        "R2": r2,
        "Best_Params": best_params,
        "Transform": "log" if use_log else "raw"
    }




# -------------------------------
# Run Full Pipeline
# -------------------------------
def run_full_pipeline(horizon_short=8, horizon_long=52):
    results = []

    SCRIPT_DIR = Path(__file__).resolve().parent
    clean_data_path = SCRIPT_DIR.parent/'data'/'processed'/'cleaned_data.csv'
    RESULTS_DIR = SCRIPT_DIR.parent /'results' 
    RESULTS_DIR.mkdir(parents=True, exist_ok=True) 

    try:
        df = pd.read_csv(clean_data_path)
        print(f"Data loaded successfully!")
    except FileNotFoundError:
        print(f"Error: One or more files were not found. Please check paths.")
        return

    # -------------------
    # Short-term models
    # -------------------
    # print("Running ARIMA and SARIMA...")
    # arima_sarima_df = arima_sarima_store_and_dept(df, horizon=horizon_short)
    # results.append(arima_sarima_df)

    # print("Running LightGBM for Store-Dept...")
    # lgbm_results = []
    # for (store, dept) in df.groupby(['Store','Dept']).groups.keys():
    #     res = lightgbm_forecast(df, store, dept, horizon=horizon_short)
    #     if res:
    #         lgbm_results.append(res)
    # print(lgbm_results)
    # results.append(pd.DataFrame(lgbm_results))

    # -------------------
    # Long-term models
    # -------------------
    print("Running Prophet + XGBoost for Chain...")
    # # Prophet (Chain)
    # prophet_chain = prophet_forecast(df, horizon=horizon_long, level="Chain")
    # results.append(pd.DataFrame([prophet_chain]))

    # XGBoost (Chain)
    xgb_chain  = tune_xgb_chain(df, horizon=20, n_trials=40)
    results.append(pd.DataFrame([xgb_chain]))

    print("Running Prophet + XGBoost for each Store...")
    prophet_store_results = []
    xgb_store_results = []
    # for stor in df['Store'].unique():
        # Prophet per Store
        # prophet_store = prophet_forecast(df, horizon=horizon_long, level="Store", store=store)
        # prophet_store_results.append(prophet_store)

        # XGBoost per Store
    xgb_store = tune_xgb_per_store(df, horizon=26)
    

    # results.append(pd.DataFrame(prophet_store_results))
    results.append(pd.DataFrame(xgb_store))

    # -------------------
    # Combine and save
    # -------------------
    final_results = pd.concat(results, ignore_index=True)
    csv_path = RESULTS_DIR / "forecast_results.csv"
    final_results.to_csv(csv_path, index=False)

    print(f"Results saved to {csv_path}")

    # -------------------
    # Plot Chain-level actual vs forecast
    # -------------------
    print("Plotting Chain-level forecasts...")

    # Prepare actual series
    chain_actual = df.groupby('Date')['Weekly_Sales'].sum().asfreq('W').fillna(0)

    # Prophet forecast (already returned inside prophet_chain['Forecast'])
    # if isinstance(prophet_chain, dict) and "Forecast" in prophet_chain:
    #     prophet_preds = prophet_chain["Forecast"].set_index("ds")["yhat"]
    # else:
    #     prophet_preds = None

    # XGBoost forecast (just RMSE/R2 before, need predictions too)
    if isinstance(xgb_chain, dict) and "Preds" in xgb_chain:
        xgb_preds = pd.Series(xgb_chain["Preds"], 
                              index=chain_actual.index[-len(xgb_chain["Preds"]):])
    else:
        xgb_preds = None

    plt.figure(figsize=(12,6))
    plt.plot(chain_actual, label="Actual", color="black")

    # if prophet_preds is not None:
    #     plt.plot(prophet_preds.index, prophet_preds.values, 
    #              label="Prophet Forecast", color="blue")

    if xgb_preds is not None:
        plt.plot(xgb_preds.index, xgb_preds.values, 
                 label="XGBoost Forecast", color="green")

    plt.title("Chain-Level Sales: Actual vs Forecast")
    plt.xlabel("Date")
    plt.ylabel("Weekly Sales")
    plt.legend()
    plt.tight_layout()

    plot_path = RESULTS_DIR / "chain_forecast_plot.png"
    plt.savefig(plot_path)
    plt.show()
    print(f"Chain-level forecast plot saved to {plot_path}")

    return final_results

if __name__ == "__main__":
    run_full_pipeline()

