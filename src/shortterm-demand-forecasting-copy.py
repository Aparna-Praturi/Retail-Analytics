import pandas as pd
import numpy as np
from pathlib import Path
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_squared_error, r2_score
from lightgbm import LGBMRegressor
import joblib
import warnings
from statsmodels.tools.sm_exceptions import ConvergenceWarning
from tqdm import tqdm
import optuna
from concurrent.futures import ThreadPoolExecutor, as_completed

warnings.simplefilter("ignore", ConvergenceWarning)
warnings.simplefilter("ignore", UserWarning)



# Helper for SARIMA objective function
def seasonal_candidates_short_term(n_train_weeks: int):
    cands = [1, 4, 8, 13]  # 1 means "no seasonality"
    if n_train_weeks >= 2 * 26:   # only if >= ~1 year of history
        cands.append(26)
    return cands

# SARIMA objective function
def sarimax_objective_short(trial, y_train_tune: pd.Series, y_val_tune: pd.Series, 
                            exog_train_tune: pd.DataFrame, exog_val_tune: pd.DataFrame):
   
    n_train_weeks = len(y_train_tune)
    s = trial.suggest_categorical("s", seasonal_candidates_short_term(n_train_weeks))

    # Non-seasonal orders
    p = trial.suggest_int("p", 0, 2)
    d = trial.suggest_int("d", 0, 1)
    q = trial.suggest_int("q", 0, 2)

    # Seasonality switch
    if s == 1:
        P = D = Q = 0
    else:
        P = trial.suggest_int("P", 0, 1)
        if s >= 26:
            D = 0
        else:
            D = trial.suggest_int("D", 0, 1)
        Q = trial.suggest_int("Q", 0, 1)

    # Transform is fixed to log for stability
    transform = "log"

    try:
        # 1. Prepare history and exog data for walk-forward loop
        history = np.log1p(y_train_tune)
        exog_history = exog_train_tune.copy()
        
        preds = []

        for t in range(len(y_val_tune)):
            # Define current training and future exog data for the next step
            current_exog_train = exog_history
            exog_future = exog_val_tune.iloc[[t]] 

            model = SARIMAX(
                history,
                exog=current_exog_train, 
                order=(p, d, q),
                seasonal_order=(P, D, Q, s),
                enforce_stationarity=False,
                enforce_invertibility=False,
            )
            fitted = model.fit(disp=False, maxiter=50)
            
            # Forecast 1 step with future exog data
            preds.append(fitted.forecast(steps=1, exog=exog_future)[0])

            # Update history for the next loop iteration
            new_obs = np.log1p(y_val_tune.iloc[t])
            history = pd.concat([history, pd.Series([new_obs], index=[y_val_tune.index[t]])])
            exog_history = pd.concat([exog_history, exog_val_tune.iloc[[t]]])

        # Inverse transform
        preds = np.expm1(preds)

        r2 = r2_score(y_val_tune, preds)
        return r2 
    except Exception:
        return -float("inf")


# SARIMAX with Optuna

def fit_sarimax_short_term(store, df, exog_cols=None, horizon=8, n_trials=25):
  
    df = df.copy().sort_index()
    series = df['Weekly_Sales'].copy()
    
    X = df[exog_cols].copy()
    
    if not isinstance(series.index, pd.DatetimeIndex):
        raise ValueError("DataFrame index must be DateTimeIndex.")
    series = series.asfreq("W-FRI").fillna(0).sort_index()
    X = X.reindex(series.index).fillna(0) # Align exog index to sales index

    if len(series) < 40:
        return None

    # 2. Data Splits (Train/Test)
    train_size = int(len(series) * 0.8)
    y_train, y_test_full = series.iloc[:train_size], series.iloc[train_size:]
    X_train, X_test_full = X.iloc[:train_size], X.iloc[train_size:]
    y_test = y_test_full.iloc[:horizon]
    X_test = X_test_full.iloc[:horizon]

    if len(y_train) < 12 or len(y_test) < 4:
        print(f" Store {store}: skipped SARIMAX/Hybrid - insufficient data (train={len(y_train)}, test={len(y_test)})")
        return None

    # 3. Inner Train/Val Split for Optuna Tuning
    split = int(len(y_train) * 0.8)
    y_train_tune, y_val_tune = y_train.iloc[:split], y_train.iloc[split:]
    X_train_tune, X_val_tune = X_train.iloc[:split], X_train.iloc[split:]


    # 4. Optuna Study
    study = optuna.create_study(direction="maximize")
    study.optimize(lambda tr: sarimax_objective_short(
        tr, y_train_tune, y_val_tune, X_train_tune, X_val_tune),
        n_trials=n_trials, show_progress_bar=False)
    
    bp = study.best_params
    s = bp["s"]
    p, d, q = bp["p"], bp["d"], bp["q"]
    transform = "log" # Transform is fixed in the objective

    # 5. Get Seasonal Parameters (Robustly)
    P = 0 if s == 1 else bp.get("P", 0) 
    Q = 0 if s == 1 else bp.get("Q", 0) 
    D = 0 if s == 1 or s >= 26 else bp.get("D", 0)


    # 6. Final Fit on Full y_train (using X_train)
    transformed_train = np.log1p(y_train)
    final_model = SARIMAX(
        transformed_train,
        exog=X_train, # Pass X_train for final fit
        order=(p, d, q),
        seasonal_order=(P, D, Q, s),
        enforce_stationarity=False,
        enforce_invertibility=False,
    )
    sarima_fit = final_model.fit(disp=False, maxiter=50)

    # 7. Walk-forward test (refit each step using X_test as future exog)
    history = transformed_train.copy()
    exog_history = X_train.copy()
    preds = []
    
    for t in range(len(y_test)):
        model = SARIMAX(
            history,
            exog=exog_history, # Current Exog data
            order=(p, d, q),
            seasonal_order=(P, D, Q, s),
            enforce_stationarity=False,
            enforce_invertibility=False,
        )
        fitted = model.fit(disp=False, maxiter=50)
        
        exog_future = X_test.iloc[[t]] # Future Exog for the next step
        preds.append(fitted.forecast(steps=1, exog=exog_future)[0])
        
        # Update history for the next iteration
        new_obs = np.log1p(y_test.iloc[t])
        history = pd.concat([history, pd.Series([new_obs], index=[y_test.index[t]])])
        exog_history = pd.concat([exog_history, X_test.iloc[[t]]])


    # Inverse transform and metrics
    preds = np.expm1(preds)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    r2 = r2_score(y_test, preds)

    results = {
        "Store": store,
        "Model": "SARIMAX", # Changed name
        "Transform": transform,
        "RMSE": rmse,
        "R2": r2,
        "Best_Params": bp,
    }
    return sarima_fit, results, (y_train, y_test, X_train, X_test) # Added X_train/X_test to return

## Helper functions for feature engineering for LightGBM

def make_lags(df, nlags = [4,8, 13], nrolling=[4,8, 13]):
    if 'y' not in df.columns:
        raise ValueError("DataFrame must contain a 'y' column for lag creation.")
    df = df.copy()
    for lag in  nlags:
        df[f'lag_{lag}'] = df['y'].shift(lag)
    for window in nrolling:
        df[f'rolling_mean_{window}'] = df['y'].rolling(window=window).mean().shift(1)
            
        # Rolling Standard Deviation (excluding current period)
        df[f'rolling_std_{window}'] = df['y'].rolling(window=window).std().shift(1)
        
    return df

def create_features(df: pd.DataFrame):
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("DataFrame index must be a DateTimeIndex.")
    df = df.copy()
    df['year'] = df.index.year
    df['month'] = df.index.month
    df['weekofyear'] = df.index.isocalendar().week.astype(int)
    df['dayofweek'] = df.index.dayofweek
    # df['quarter'] = df.index.quarter
    # df['is_month_start'] = df.index.is_month_start.astype(int)
    # df['is_month_end'] = df.index.is_month_end.astype(int)
    # df['is_quarter_start'] = df.index.is_quarter_start.astype(int)
    # df['is_quarter_end'] = df.index.is_quarter_end.astype(int)
    df['is_year_start'] = df.index.is_year_start.astype(int)
    df['is_year_end'] = df.index.is_year_end.astype(int)
    df['time_idx'] = (df.index - df.index.min()).days
    return df


def lgbm_objective(trial, X_res, y_res, n_splits=5):
  
    
    # 1. Suggest Hyperparameters
    param = {
        'objective': 'regression',
        'metric': 'rmse',
        'n_estimators': trial.suggest_int('n_estimators', 100, 500),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
        'feature_fraction': trial.suggest_float('feature_fraction', 0.5, 1.0),
        'bagging_fraction': trial.suggest_float('bagging_fraction', 0.5, 1.0),
        'bagging_freq': trial.suggest_int('bagging_freq', 1, 7),
        'lambda_l1': trial.suggest_float('lambda_l1', 1e-8, 10.0, log=True),
        'lambda_l2': trial.suggest_float('lambda_l2', 1e-8, 10.0, log=True),
        'num_leaves': trial.suggest_int('num_leaves', 2, 64),
        'verbose': -1,
        'n_jobs': -1,
        'seed': 42 
    }

    
    # Use TimeSeriesSplit to simulate sequential residual prediction validation
    from sklearn.model_selection import TimeSeriesSplit
    tscv = TimeSeriesSplit(n_splits=n_splits)
    def rolling_folds_index(n, val_size=8, min_train=40, step=8):
     # yields (train_idx, val_idx) for rolling-origin CV
        start = min_train
        while start + val_size <= n:
            train_idx = np.arange(0, start)
            val_idx   = np.arange(start, start + val_size)
            yield train_idx, val_idx
            start += step
        
    r2_list = []
    rmse_list=[]
    n = len(X_res)  
    
    for train_index, val_index in rolling_folds_index(n, val_size=8, min_train=40, step=8):
        X_train, X_val = X_res.iloc[train_index], X_res.iloc[val_index]
        y_train, y_val = y_res.iloc[train_index], y_res.iloc[val_index]
        
        # Check for sufficient data
        if len(y_val) == 0:
            continue
        try:
            model = LGBMRegressor(**param)
            model.fit(X_train, y_train, 
                      eval_set=[(X_val, y_val)],
                    #   early_stopping_rounds=50, # Early stop to prevent overfitting
                      eval_metric='rmse',
                      callbacks=[optuna.integration.LightGBMPruningCallback(trial, 'rmse')]
                      )
            
            preds = model.predict(X_val)
            rmse = np.sqrt(mean_squared_error(y_val, preds))
            r2 = r2_score(y_val, preds)
            r2_list.append(r2)
            rmse_list.append[rmse]
        
        except Exception as e:
            print(f"Trial failed: {e}")
            return float("inf") 
    return np.mean(rmse_list) if r2_list else float("inf")


def fit_lgbm_direct_short_term(store, df, exog_cols=None, horizon=8, n_trials=35):
  
    df = df.copy().sort_index()
    series = df['Weekly_Sales'].copy()
    
    #  Prepare Data and Features
    series_df = pd.DataFrame({'y': series})
    
    # Add calendar features
    X_feats = create_features(series_df) 
    
    # Add lagged sales features
    X_feats = make_lags(X_feats, nlags = [1,2,4,8], nrolling=[4,8])
    
    # Add exogenous features
    if exog_cols:
        X_exog = df[exog_cols].copy()
        X_feats = X_feats.join(X_exog, how='left')

    # Drop rows with NaN 
    X_feats = X_feats.dropna()
    y = X_feats['y'].copy()
    X = X_feats.drop(columns=['y'])

    #  Data Splits
    train_size = int(len(y) * 0.8)
    y_train, y_test = y.iloc[:train_size], y.iloc[train_size:]
    X_train, X_test = X.iloc[:train_size], X.iloc[train_size:]

    if len(y_train) < 10 or len(y_test) < 4:
        print(f"Store {store}: skipped Direct LGBM - insufficient data.")
        return None

 
    #  Optuna Study 
    study = optuna.create_study(direction="minimize", study_name=f"lgbm_tune_direct_store_{store}")
    study.optimize(lambda trial: lgbm_objective(trial, X_train, y_train), 
                   n_trials=n_trials, show_progress_bar=False)
    
    bp = study.best_params
    bp.update({'objective': 'regression', 'n_jobs': -1, 'seed': 42, 'verbose': -1})

    #  Final Fit on Full Training Data
    lgbm_fit = LGBMRegressor(**bp)
    lgbm_fit.fit(X_train, y_train)

    #  Prediction on Test Set 
    preds = lgbm_fit.predict(X_test)

    #  Metrics
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    r2 = r2_score(y_test, preds)

    results = {
        "Store": store,
        "Model": "LGBM_Direct",
        "Transform": "none",
        "RMSE": rmse,
        "R2": r2,
        "Best_Params": bp,
        "Trained_Model": lgbm_fit
    }
    return results


# Naive baseline 

def naive_forecast(store, df, horizon=8, season_length=52):
    series = df['Weekly_Sales'].asfreq('W-FRI').fillna(0).sort_index()
    train_size = int(len(series) * 0.8)
    y_train, y_test_full = series.iloc[:train_size], series.iloc[train_size:]

    if len(y_test_full) < horizon or len(y_train) <= season_length:
        return None

    y_test = y_test_full.iloc[:horizon]
    preds = y_train.iloc[-season_length:][-horizon:].values

    rmse = np.sqrt(mean_squared_error(y_test, preds))
    r2 = r2_score(y_test, preds)
    return {"Store": store, "Model": "Naive", "RMSE": rmse, "R2": r2, "Level": "Store"}




# Run all models for one store 
def run_all_models_for_store(store, df_store, exog_cols=None, horizon=8):
    results = []  
    try:
        # 1. Run the SARIMAX fit
        out = fit_sarimax_short_term(store, df_store, exog_cols=exog_cols, horizon=horizon, n_trials=10)
        
        # Check if the fit returned data (e.g., if len(series) < 40 returns None)
        if out is None:
            raise ValueError("SARIMAX fit skipped due to insufficient data.")
        
        sarima_fit, sarima_metrics, data_splits = out 
    
        #  Append SARIMAX metrics 
        sarima_metrics["Level"] = "Store"
        sarima_metrics["Trained_Model"] = sarima_fit
        results.append(sarima_metrics) 

        # # 4. Run Hybrid model (passing the full result tuple 'out')
        # # The hybrid model needs to be corrected as well (see next section)
        # hybrid_res = train_hybrid_model(store, df_store, exog_cols=exog_cols, 
        #                                 horizon=horizon, sarimax_result=out) # Renamed arg
        # if hybrid_res is not None:
        #     results.append(hybrid_res)
    
    except Exception as e:
         # This will catch the ValueError if out is None, or any other SARIMAX error
        print(f" SARIMAX/Hybrid failed for Store {store}: {e}")

    try:
        lgbm_res = fit_lgbm_direct_short_term(store, df_store, exog_cols=exog_cols, horizon=8, n_trials=30)
        if lgbm_res is not None:
            results.append(lgbm_res)
    except Exception as e:
        print(f" lgbm forecast failed for Store {store}: {e}")


    try:
        naive_res = naive_forecast(store, df_store, horizon=horizon)
        if naive_res is not None:
            results.append(naive_res)
    except Exception as e:
        print(f" Naive forecast failed for Store {store}: {e}")

    return results


# Best model selector 
def pick_best_model_per_store(results_df):
    def pick_best(group):
        group = group.sort_values(by=['R2', 'RMSE'], ascending=[False, True])
        return group.iloc[0]
    best_models = results_df.groupby('Store', group_keys=False).apply(pick_best)
    return best_models.reset_index(drop=True)

# Save Best Model Helper
def save_best_model(model_obj, level, store, model_name):
    model_dir = Path("saved_models") /"short-term"/ level
    model_dir.mkdir(parents=True, exist_ok=True)
    filename = f'Store_{int(store)}_{model_name}_.pkl'
    joblib.dump(model_obj, model_dir / filename)



# Short-term pipeline 
def run_shortterm_pipeline(horizon=8, n_jobs=6):
    SCRIPT_DIR = Path(__file__).resolve().parent
    clean_data_path = SCRIPT_DIR.parent / 'data' / 'processed' / 'cleaned_data.csv'
    df = pd.read_csv(clean_data_path, parse_dates=['Date'])
    print(f" Data loaded successfully!")

    # ensure exog columns exist; create zeros if missing
    exog_cols= ['IsHoliday', 'Total_MarkDown']
    for c in exog_cols:
        if c not in df.columns:
            df[c] = 0.0

    df = df[['Store','Date','Weekly_Sales'] + exog_cols].copy()
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.set_index('Date')

    print("\n Preparing store frames...")
    tasks = []
    # (loop all stores; here you had a single store=10 for testing)
    # for store in df['Store'].unique():

    for store in df['Store'].unique():
        
        agg = df[df['Store'] == store].drop(columns=['Store'])
        
        agg_map = {'Weekly_Sales': 'sum'}
        
        valid_exog_cols = [c for c in exog_cols if c in agg.columns]
        agg_map.update({c: 'mean' for c in valid_exog_cols})
        
        # 3. Resample the store's data weekly, aggregate, and fill missing weeks with 0
        #    (Fixed: using 'agg' instead of the undefined variable 'g')
        df_store = agg.resample('W-FRI').agg(agg_map).fillna(0)
        
        # Optional check: uncomment if you want to skip stores with very little history
        if len(df_store) < horizon * 3:
            print(f"Skipping Store {store}: insufficient data points ({len(df_store)} weeks)")
            continue
            
        # 4. Append the prepared data and store ID to the task list
        tasks.append((store, df_store))

    print(f"\n Running SARIMA (no exog) + Hybrid(LGBM with exog) + Naive for {len(tasks)} stores (parallel={n_jobs})...")
    all_results = []
    with ThreadPoolExecutor(max_workers=n_jobs) as executor:
        futures = [executor.submit(run_all_models_for_store, store, df_store, exog_cols, horizon)
                   for store, df_store in tasks]
        for future in tqdm(as_completed(futures), total=len(futures), desc="Model fits"):
            try:
                store_results = future.result()
                if store_results:
                    all_results.extend(store_results)
            except Exception as e:
                print(f" Parallel task failed: {e}")

    results_df = pd.DataFrame(all_results)

    print("\n Selecting best model per store...")
    best_models_df = pick_best_model_per_store(results_df)

    output_path = Path("results") / "shortterm_forecast_results_full.csv"
    output_path.parent.mkdir(exist_ok=True)
    results_df.to_csv(output_path, index=False)

    best_output_path = Path("results") / "shortterm_best_models.csv"
    best_models_df.to_csv(best_output_path, index=False)
    print(f" Best models saved to {best_output_path}")

    print("\n Saving best models per store...")
    for _, row in best_models_df.iterrows():
        if row["Model"] == "SARIMA":
            save_best_model(row.get('Trained_Model'), level='Store', store=row['Store'], model_name='SARIMA')
        elif row["Model"] == "LGBM_Direct":
            save_best_model(row.get('Trained_Model'), level='Store', store=row['Store'], model_name='LGBM')  
    print(" Best models saved successfully.")

# Run Script
if __name__ == "__main__":
    run_shortterm_pipeline(horizon=8, n_jobs=6)
