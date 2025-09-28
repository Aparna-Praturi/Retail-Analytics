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


# ============================================================
# SARIMA Optuna Objective (with walk-forward)
# ============================================================
import numpy as np
import pandas as pd
import optuna
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_squared_error, r2_score
import warnings

warnings.filterwarnings("ignore")


# ============================================================
# 1️⃣ SARIMA Objective Function (for Optuna tuning)
# ============================================================
import numpy as np
import pandas as pd
import optuna
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_squared_error, r2_score
import warnings
warnings.filterwarnings("ignore")


def seasonal_candidates_short_term(n_train_weeks: int):
    cands = [1, 4, 8, 13]  # 1 means "no seasonality"
    if n_train_weeks >= 2 * 26:   # only if >= ~1 year of history
        cands.append(26)
    return cands


def sarima_objective_short(trial, y_train_tune: pd.Series, y_val_tune: pd.Series):
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
        # Guard against fragile long-lag differencing
        if s >= 26:
            D = 0
        else:
            D = trial.suggest_int("D", 0, 1)
        Q = trial.suggest_int("Q", 0, 1)

    transform = trial.suggest_categorical("transform", ["raw", "log"])

    try:
        history = np.log1p(y_train_tune) if transform == "log" else y_train_tune.copy()
        preds = []

        for t in range(len(y_val_tune)):
            model = SARIMAX(
                history,
                order=(p, d, q),
                seasonal_order=(P, D, Q, s),
                enforce_stationarity=False,
                enforce_invertibility=False,
            )
            fitted = model.fit(disp=False, maxiter=50)
            preds.append(fitted.forecast(steps=1)[0])

            new_obs = np.log1p(y_val_tune.iloc[t]) if transform == "log" else y_val_tune.iloc[t]
            history = pd.concat([history, pd.Series([new_obs], index=[y_val_tune.index[t]])])

        if transform == "log":
            preds = np.expm1(preds)

        rmse = np.sqrt(mean_squared_error(y_val_tune, preds))
        return rmse  # minimize RMSE
    except Exception:
        return float("inf")


def fit_sarima_short_term(store, series: pd.Series, horizon=8, n_trials=10):
    """
    Short-horizon (8w) SARIMA tuning & walk-forward test evaluation.
    Expects `series` weekly, numeric, DateTimeIndex, sorted.
    """
    series = series.copy()
    # Ensure weekly grid; change 'W-FRI' if your week ends Sunday etc.
    if not isinstance(series.index, pd.DatetimeIndex):
        raise ValueError("Series index must be DateTimeIndex.")
    series = series.asfreq("W-FRI").fillna(0).sort_index()

    if len(series) < 40:  # need some history
        return None

    train_size = int(len(series) * 0.8)
    y_train, y_test = series.iloc[:train_size], series.iloc[train_size:]

    # ---- Guard clause: avoid short series ----
    if len(y_train) < 12 or len(y_test) < 4:
        print(f"⚠️ Store {store}: skipped SARIMA/Hybrid - insufficient data "
              f"(train={len(y_train)}, test={len(y_test)})")
        return None, None, None

    # if len(y_test) < horizon:
    #     return None

    # Train/val split inside train
    split = int(len(y_train) * 0.8)
    y_train_tune, y_val_tune = y_train.iloc[:split], y_train.iloc[split:]

    study = optuna.create_study(direction="minimize")
    study.optimize(lambda tr: sarima_objective_short(tr, y_train_tune, y_val_tune),
                   n_trials=n_trials, show_progress_bar=False)

    bp = study.best_params
    s = bp["s"]
    p, d, q = bp["p"], bp["d"], bp["q"]
    transform = bp["transform"]

    if s == 1:
        P = D = Q = 0
    else:
        P = bp["P"]
        Q = bp["Q"]
        D = 0 if s >= 26 else bp["D"]

    # =======================
    #  Final Fit on Training Data
    # =======================
    transformed_train = np.log1p(y_train) if transform == "log" else y_train.copy()
    final_model = SARIMAX(
        transformed_train,
        order=(p, d, q),
        seasonal_order=(P, D, Q, s),
        enforce_stationarity=False,
        enforce_invertibility=False,
    )
    sarima_fit = final_model.fit(disp=False, maxiter=50)

    # =======================
    # ✅ Test Forecast Evaluation
    # =======================
    history = np.log1p(y_train) if transform == "log" else y_train.copy()
    preds = []
    for t in range(len(y_test)):
        model = SARIMAX(
            history,
            order=(p, d, q),
            seasonal_order=(P, D, Q, s),
            enforce_stationarity=False,
            enforce_invertibility=False,
        )
        fitted = model.fit(disp=False, maxiter=50)
        preds.append(fitted.forecast(steps=1)[0])
        new_obs = np.log1p(y_test.iloc[t]) if transform == "log" else y_test.iloc[t]
        history = pd.concat([history, pd.Series([new_obs], index=[y_test.index[t]])])

    if transform == "log":
        preds = np.expm1(preds)

    rmse = np.sqrt(mean_squared_error(y_test, preds))
    r2 = r2_score(y_test, preds)

   
    results = {
        "Store": store,
        "Model": "SARIMA",
        "Transform": transform,
        "RMSE": rmse,
        "R2": r2,
        "Best_Params": bp,
    }

    return sarima_fit, results, (y_train, y_test)

def make_lags(df: pd.DataFrame, nlags: int = 8):
    """
    Create lag features for a univariate time series.
    Expects a DataFrame with a DateTimeIndex and a 'y' column.
    Adds lag_1, lag_2, ..., lag_nlags columns.
    """
    df = df.copy()
    if 'y' not in df.columns:
        raise ValueError("DataFrame must contain a 'y' column for lag creation.")

    for lag in range(1, nlags + 1):
        df[f'lag_{lag}'] = df['y'].shift(lag)

    return df

def create_features(df: pd.DataFrame):
    """
    Create calendar-based and simple time series features for forecasting.
    Expects a DataFrame with a DateTimeIndex and a 'y' column (target or residuals).
    Returns the DataFrame with added feature columns.
    """

    df = df.copy()
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("DataFrame index must be a DateTimeIndex.")

    # --- Calendar/time features ---
    df['year'] = df.index.year
    df['month'] = df.index.month
    df['weekofyear'] = df.index.isocalendar().week.astype(int)
    df['dayofweek'] = df.index.dayofweek  # Monday=0, Sunday=6
    df['quarter'] = df.index.quarter

    # --- Seasonal indicators ---
    df['is_month_start'] = df.index.is_month_start.astype(int)
    df['is_month_end'] = df.index.is_month_end.astype(int)
    df['is_quarter_start'] = df.index.is_quarter_start.astype(int)
    df['is_quarter_end'] = df.index.is_quarter_end.astype(int)
    df['is_year_start'] = df.index.is_year_start.astype(int)
    df['is_year_end'] = df.index.is_year_end.astype(int)

    # --- Trend feature ---
    df['time_idx'] = (df.index - df.index.min()).days
    

    return df


def train_hybrid_model(store, series, horizon=8):

    result = fit_sarima_short_term(store, series, horizon=horizon)
    if result is None:
        return {"Store": store, "Model": "Naive", "R2": np.nan, "Reason": "sarima_failed"}

    sarima_fit, sarima_results, (y_train, y_test) = result
    # Compute residuals
    residuals = y_train - (np.expm1(sarima_fit.fittedvalues) 
                           if sarima_results["Transform"] == "log" 
                           else sarima_fit.fittedvalues)

    df = pd.DataFrame({'y': residuals}, index=y_train.index)
    df = create_features(make_lags(df)).dropna()
    X, y_res = df.drop(columns=['y']), df['y']

    lgb_model = LGBMRegressor(n_estimators=300, learning_rate=0.05, num_leaves=31)
    lgb_model.fit(X, y_res)

    # --- Hybrid forecast ---
    sarima_fc = sarima_fit.get_forecast(steps=len(y_test)).predicted_mean
    if sarima_results["Transform"] == "log":
        sarima_fc = np.expm1(sarima_fc)

    hybrid_preds = []
    history = series.copy()

    for step, t in enumerate(y_test.index):
        df_step = pd.DataFrame({'y': history}, index=history.index)
        df_step = create_features(make_lags(df_step))
        X_step = df_step.iloc[[-1]].drop(columns=['y'])
        res_fc = lgb_model.predict(X_step)[0] if not X_step.isnull().any().any() else 0
        hybrid_preds.append(sarima_fc.iloc[step] + res_fc)
        history.loc[t] = y_test.iloc[step]

    hybrid_r2 = r2_score(y_test, hybrid_preds)

    # return {
    #     "Store": store,
    #     "Model": "Hybrid" if hybrid_r2 > sarima_results["R2"] else "SARIMA",
    #     "SARIMA_R2": sarima_results["R2"],
    #     "Hybrid_R2": hybrid_r2,
    #     "Best_Params": sarima_results["Best_Params"]
    # }

    
    return {
        "Store": store,
        "Model": "Hybrid",
        "Hybrid_R2": hybrid_r2,
        "SARIMA_R2": sarima_results["R2"],
        "SARIMA_RMSE": sarima_results["RMSE"],
        "Best_Params": sarima_results["Best_Params"],
        "Trained_SARIMA": sarima_fit,
        "Trained_LGBM": lgb_model
    }

def naive_forecast(store, series, horizon=8):
    """
    Naïve forecasting baseline: predicts the last observed value for all future steps.
    Evaluated using walk-forward style.
    """
    series = series.asfreq('W-FRI').fillna(0).sort_index()
    train_size = int(len(series) * 0.8)
    y_train, y_test = series.iloc[:train_size], series.iloc[train_size:]

    if len(y_test) < horizon or len(y_train) == 0:
        return None

    last_value = y_train.iloc[-1]
    preds = np.repeat(last_value, len(y_test))

    rmse = np.sqrt(mean_squared_error(y_test, preds))
    r2 = r2_score(y_test, preds)

    return {
        "Store": store,
        "Model": "Naive",
        "RMSE": rmse,
        "R2": r2,
        "Level": "Store"
    }


def run_all_models_for_store(store, series, horizon=8):
    """
    Run SARIMA, Hybrid, and Naive forecasting for a single store.
    Returns a list of results dicts.
    """
    results = []

    # --- SARIMA ---
    try:
        sarima_fit, sarima_metrics, (y_train, y_test) = fit_sarima_short_term(store, series, horizon=horizon)
        if sarima_fit is not None:
            sarima_metrics["Level"] = "Store"
            sarima_metrics["Trained_Model"] = sarima_fit
            results.append(sarima_metrics)

            # --- Hybrid ---
            hybrid_res = train_hybrid_model(store, series, horizon=horizon)
            if hybrid_res is not None:
                results.append(hybrid_res)
    except Exception as e:
        print(f"⚠️ SARIMA/Hybrid failed for Store {store}: {e}")

    # --- Naive ---
    try:
        naive_res = naive_forecast(store, series, horizon=horizon)
        if naive_res is not None:
            results.append(naive_res)
    except Exception as e:
        print(f"⚠️ Naive forecast failed for Store {store}: {e}")

    return results

def pick_best_model_per_store(results_df):
    """
    For each Store, pick the model with the best performance.
    Prioritize highest R²; if missing, use lowest RMSE.
    """
    def pick_best(group):
        # Sort by R² (descending), then RMSE (ascending)
        group = group.sort_values(by=['R2', 'RMSE'], ascending=[False, True])
        return group.iloc[0]  # best row

    best_models = results_df.groupby('Store', group_keys=False).apply(pick_best)
    return best_models.reset_index(drop=True)


def run_shortterm_pipeline(horizon=8, n_jobs=6):
    SCRIPT_DIR = Path(__file__).resolve().parent
    clean_data_path = SCRIPT_DIR.parent / 'data' / 'processed' / 'cleaned_data.csv'
    df = pd.read_csv(clean_data_path, parse_dates=['Date'])
    print(f"✅ Data loaded successfully!")

    df = df[['Store', 'Date', 'Dept', 'Weekly_Sales']].copy()
    df['Date'] = pd.to_datetime(df['Date'])

    # --- Prepare tasks ---
    print("\n📝 Preparing store series...")
    tasks = []
    # for store in tqdm(df['Store'].unique()):
    store =10
    series = (
        df[df['Store'] == store]
        .groupby('Date')['Weekly_Sales']
        .sum()
        .sort_index()
    )
    series = series.asfreq('W-FRI')

    # if len(series) < horizon * 3:
    #     continue
    tasks.append((store, series))

    # --- Run all models in parallel ---
    print(f"\n🚀 Running SARIMA + Hybrid + Naive for {len(tasks)} stores (parallel={n_jobs})...")
    all_results = []
    with ThreadPoolExecutor(max_workers=n_jobs) as executor:
        futures = [executor.submit(run_all_models_for_store, store, series, horizon) 
                   for store, series in tasks]

        for future in tqdm(as_completed(futures), total=len(futures), desc="Model fits"):
            try:
                store_results = future.result()
                if store_results:
                    all_results.extend(store_results)
            except Exception as e:
                print(f"⚠️ Parallel task failed: {e}")

    # --- Convert to DataFrame ---
    results_df = pd.DataFrame(all_results)

     # --- Convert to DataFrame ---
    results_df = pd.DataFrame(all_results)

    # --- Pick best model per store ---
    print("\n🏆 Selecting best model per store...")
    best_models_df = pick_best_model_per_store(results_df)

    # --- Save results ---
    output_path = Path("results") / "shortterm_forecast_results_full.csv"
    output_path.parent.mkdir(exist_ok=True)
    results_df.to_csv(output_path, index=False)

    best_output_path = Path("results") / "shortterm_best_models.csv"
    best_models_df.to_csv(best_output_path, index=False)
    print(f"📄 Best models saved to {best_output_path}")

    # --- Save only the best models ---
    print("\n💾 Saving best models per store...")
    for _, row in best_models_df.iterrows():
        if row["Model"] == "SARIMA":
            save_best_model(row['Trained_Model'], level='Store', store=row['Store'], model_name='SARIMA')
        elif row["Model"] == "Hybrid":
            save_best_model(row['Trained_SARIMA'], level='Store', store=row['Store'], model_name='SARIMA')
            save_best_model(row['Trained_LGBM'], level='Store', store=row['Store'], model_name='Hybrid_LGBM')
    print("✅ Best models saved successfully.")

    # # --- Save results ---
    # output_path = Path("results") / "shortterm_forecast_results_full.csv"
    # output_path.parent.mkdir(exist_ok=True)
    # results_df.to_csv(output_path, index=False)
    # print(f"\n📄 Results (SARIMA + Hybrid + Naive) saved to {output_path}")


    # print("\n✅ Pipeline completed successfully!")
    # return results_df



# def run_sarima_parallel(tasks, n_jobs=4):
#     results = []
#     with ThreadPoolExecutor(max_workers=n_jobs) as executor:
#         futures = [executor.submit(fit_sarima_short_term, store, series) 
#                    for store, series in tasks]

#         for future in tqdm(as_completed(futures), total=len(futures), desc="SARIMA fits"):
#             try:
#                 res = future.result()
#                 if res is not None:
#                     results.append(res)
#             except Exception as e:
#                 print(f"⚠️ SARIMA fit failed: {e}")
#     return results



# ============================================================
# Save Best Model Helper
# ============================================================
def save_best_model(model_obj, level, store, model_name):
    model_dir = Path("saved_models") /"short-term"/ level
    model_dir.mkdir(parents=True, exist_ok=True)
    filename = f"{model_name}_{'Chain' if pd.isna(store) else f'Store_{int(store)}'}.pkl"
    joblib.dump(model_obj, model_dir / filename)


# # ============================================================
# # Final Short-term Pipeline
# # ============================================================
# def run_shortterm_pipeline(horizon=8, n_jobs=4):
#     SCRIPT_DIR = Path(__file__).resolve().parent
#     clean_data_path = SCRIPT_DIR.parent / 'data' / 'processed' / 'cleaned_data.csv'
#     df = pd.read_csv(clean_data_path, parse_dates=['Date'])
#     print(f"✅ Data loaded successfully!")

#     results = []
#     df = df[['Store','Date','Dept', 'Weekly_Sales']].copy()
#     df['Date'] = pd.to_datetime(df['Date'])

#     # # --- SARIMA per Store ---
#     print("\n🏪 Running Store-level SARIMA (Parallel + Tuning)...")
#     tasks = []
#     for store in tqdm(df['Store'].unique()):
    
#         series = (
#             df[df['Store']==store]
#             .groupby('Date')['Weekly_Sales']
#             .sum()
#             .sort_index()
#         )
#         series = series.asfreq('W-FRI') 
#         # if len(series) < horizon*3:
#         #     continue
#         tasks.append((store, series))
#     results = run_sarima_parallel(tasks, n_jobs=4)  
   

#     # --- LightGBM per Store-Dept ---
#     # print("\n📊 Running LightGBM per Store–Dept (Tuning)...")

#     # for (store, dept), _ in df.groupby(['Store','Dept']):
#     #     res = lightgbm_forecast(df, store, dept, horizon=horizon)
#     #     if res:
#     #         results.append(res)

#     results_df = pd.DataFrame(results)
# # 
#     # # Pick best model per group
#     # def pick_best(group):
#     #     group = group.sort_values(by=['R2','RMSE'], ascending=[False,True])
#     #     group['Best_Model'] = [True]+[False]*(len(group)-1)
#     #     return group

#     # results_df = results_df.groupby(['Level','Store'], dropna=False).apply(pick_best).reset_index(drop=True)

#     # Save best models
#     # print("\n💾 Saving best models...")
#     # best_rows = results_df[results_df['Best_Model']]
#     # for _, row in best_rows.iterrows():
#     #     save_best_model(row['Trained_Model'], row['Level'], row['Store'], row['Model'])

#     # Save results CSV
#     output_path = Path("results") / "shortterm_forecast_results_tuned.csv"
#     output_path.parent.mkdir(exist_ok=True)
#     results_df.to_csv(output_path, index=False)
#     print(f"\n📄 Results saved to {output_path}")

#     return results_df


# ============================================================
# Run Script
# ============================================================
if __name__ == "__main__":
    run_shortterm_pipeline(horizon=8, n_jobs=6)
