import pandas as pd
import numpy as np
from pathlib import Path
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_squared_error, r2_score
from xgboost import XGBRegressor
from prophet import Prophet
import warnings
from tqdm import tqdm
import optuna
import json
import joblib

warnings.filterwarnings("ignore")

RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)



# Walk-forward validation for XGBoost

def walk_forward_validation(X, y, params, use_log=True, horizon=20):
    n_total = len(y)
    window_size = int(n_total * 0.7)
    rmse_scores = []
    r2_scores = []
    for start in range(window_size, n_total - horizon, horizon):
        X_train, y_train = X.iloc[:start], y.iloc[:start]
        X_val, y_val = X.iloc[start:start+horizon], y.iloc[start:start+horizon]
        y_train_model = np.log1p(y_train) if use_log else y_train
        model = XGBRegressor(**params)
        model.fit(X_train, y_train_model, verbose=False)
        y_pred = model.predict(X_val)
        if use_log:
            y_pred = np.expm1(y_pred)
        rmse_scores.append(np.sqrt(mean_squared_error(y_val, y_pred)))
        r2_scores.append(r2_score(y_val, y_pred))
    return float(np.mean(r2_scores)), float(np.mean(rmse_scores))



# Tuned XGBoost Forecast 

def xgb_forecast_tuned(df, horizon=20, n_trials=30, level="Store", store=None):
    df = df[['Store','Date','Weekly_Sales']].copy()

    if level == 'Chain':
        group = df.copy()
    elif level == 'Store':
        group = df[df['Store']==store].copy()
    else:
        return None

    group["date"] = pd.to_datetime(group["Date"])
    group = group.groupby("date")["Weekly_Sales"].sum().reset_index().sort_values("date")
    group = group.set_index("date")

    # Feature engineering
    group["lag_1"] = group["Weekly_Sales"].shift(1)
    group["lag_2"] = group["Weekly_Sales"].shift(2)
    group["lag_52"] = group["Weekly_Sales"].shift(52)
    for w in [4,12,52]:
        group[f"roll_mean_{w}"] = group["Weekly_Sales"].shift(1).rolling(w).mean()
    group["weekofyear"] = group.index.isocalendar().week.astype(int)
    group["month"] = group.index.month
    group["year"] = group.index.year

    group = group.dropna()
    X = group.drop(columns=["Weekly_Sales"])
    y = group["Weekly_Sales"]

    # Optuna objective
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
        mean_r2, _ = walk_forward_validation(X, y, params, use_log=use_log, horizon=horizon)
        return mean_r2

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials, n_jobs=-1)
    best_params = study.best_params
    use_log = best_params.pop("use_log")

    # Final evaluation
    train_size = int(len(X)*0.8)
    X_train, X_test = X.iloc[:train_size], X.iloc[train_size:]
    y_train, y_test = y.iloc[:train_size], y.iloc[train_size:]
    y_train_final = np.log1p(y_train) if use_log else y_train

    model = XGBRegressor(**best_params)
    model.fit(X_train, y_train_final)
    y_pred = model.predict(X_test)
    if use_log:
        y_pred = np.expm1(y_pred)

    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    return {"Level": level, "Store": store, "Model": "XGBoost (Tuned)", 
            "RMSE": rmse, "R2": r2, "Trained_model":model, "Date":y_test.index,
        "Actual":y_test,
        "Forecast":y_pred,
        "x_train":y_train.index,
        "y_train":y_train
    }



# Tuned Prophet Forecast 

def prophet_forecast_tuned(df, horizon=20, n_trials=30, level="Store", store=None):
    df = df[['Store','Date','Weekly_Sales']].copy()

    if level == "Chain":
        data = df.copy()
    elif level == "Store":
        data = df[df['Store']==store].copy()
    else:
        return None

    agg = data.groupby('Date')['Weekly_Sales'].sum().reset_index()
    agg.rename(columns={'Date': 'ds', 'Weekly_Sales': 'y'}, inplace=True)
    agg['ds'] = pd.to_datetime(agg['ds'])
    agg = agg.set_index('ds').asfreq('W-FRI').rename_axis('ds').reset_index().sort_values('ds')

    # log transform
    agg['y'] = np.log1p(agg['y'])

    train_size = int(len(agg) * 0.8)
    train, test = agg.iloc[:train_size].copy(), agg.iloc[train_size:].copy()

    def objective(trial):
        params = {
            "changepoint_prior_scale": trial.suggest_float("changepoint_prior_scale", 0.001, 0.5, log=True),
            "seasonality_prior_scale": trial.suggest_float("seasonality_prior_scale", 0.01, 10.0, log=True),
            "seasonality_mode": trial.suggest_categorical("seasonality_mode", ["additive", "multiplicative"]),
            "n_changepoints": trial.suggest_int("n_changepoints", 5, 50),
        }
        m = Prophet(weekly_seasonality=True, yearly_seasonality=True, **params)
        m.add_country_holidays('US')  
        m.fit(train)
        future = m.make_future_dataframe(periods=len(test), freq='W-FRI')
        forecast = m.predict(future)
        preds = forecast.iloc[-len(test):]['yhat'].values
        # invert log
        preds = np.expm1(preds); y_true = np.expm1(test['y'].values)
        return r2_score(y_true, preds)

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False,n_jobs=-1)
    best_params = study.best_params

    # Final model
    m_best = Prophet(weekly_seasonality=True, yearly_seasonality=True, **best_params)
    m_best.fit(train)
    future = m_best.make_future_dataframe(periods=len(test), freq='W')
    forecast = m_best.predict(future)

    preds = forecast.iloc[-len(test):]['yhat'].values
    y_true = test['y'].values
    rmse = np.sqrt(mean_squared_error(y_true, preds))
    r2 = r2_score(y_true, preds)

    return {"Level": level, "Store": store, "Model": "Prophet (Tuned)", 
            "RMSE": rmse, "R2": r2, "Trained_model":m_best, "Date":test['ds'],
        "Actual":y_true,
        "Forecast":preds
    }

# Naive baseline 

def naive_forecast(store, df, horizon=20, season_length=52):
    df['Date'] = pd.to_datetime(df['Date'])  
    df = df.set_index('Date')              

    series = (
        df['Weekly_Sales']
        .resample('W-FRI')
        .sum()
        .fillna(0)
        .sort_index()
)

    train_size = int(len(series) * 0.8)
    y_train, y_test_full = series.iloc[:train_size], series.iloc[train_size:]

    if len(y_test_full) < horizon or len(y_train) < 1:
        return None  # not enough data

    y_test = y_test_full.iloc[:horizon]

    # adjust season_length if training set is too short
    eff_season_length = min(season_length, len(y_train))

    # naive seasonal forecast OR last value if too short
    preds = y_train.iloc[-eff_season_length:][-horizon:].values
    if len(preds) < len(y_test):
        # pad by repeating last value
        preds = np.pad(preds, (len(y_test)-len(preds), 0), mode='edge')

    rmse = np.sqrt(mean_squared_error(y_test, preds))
    r2 = r2_score(y_test, preds)

    return {
        "Store": store,
        "Model": "Naive",
        "RMSE": rmse,
        "R2": r2,
        "Level": "Store",
        "Date": y_test.index,
        "Actual": y_test,
        "Forecast": preds,
        "x_train": y_train.index,
        "y_train": y_train
    }

 

# Helper to save model

def save_best_model(model_obj, level, store, model_name):
    """
    Saves the trained model to a structured folder for later use.
    
    """
    SCRIPT_DIR = Path(__file__).resolve().parent
    model_dir = SCRIPT_DIR.parent / "saved_models" / level
    model_dir.mkdir(parents=True, exist_ok=True)

    if pd.isna(store):  # Chain level
        filename = f"_Chain_{model_name}.pkl"
    else:
        filename = f"Store_{int(store)}_{model_name}_.pkl"

    model_path = model_dir / filename
    joblib.dump(model_obj, model_path)
    return model_path



# Final Pipeline with Saving

def run_full_pipeline(horizon_long=20):
    SCRIPT_DIR = Path(__file__).resolve().parent
    clean_data_path = SCRIPT_DIR.parent / 'data' / 'processed' / 'cleaned_data.csv'

    df = pd.read_csv(clean_data_path)
    print(f" Data loaded successfully!")

    results = []

  
    # Chain level models
 
    print("\n Running CHAIN level models...")
    results.append(xgb_forecast_tuned(df, horizon=horizon_long, level="Chain"))
    results.append(prophet_forecast_tuned(df, horizon=horizon_long, level="Chain"))
    
    
    # Store level models
   
    print("\n Running STORE level models...")
    for store in tqdm(df['Store'].unique()):
        results.append(xgb_forecast_tuned(df, horizon=horizon_long, level="Store", store=store))
        results.append(prophet_forecast_tuned(df, horizon=horizon_long, level="Store", store=store))
        results.append(naive_forecast(store, df, horizon=20, season_length=52))
    # # Convert to DataFrame
    final_results = pd.DataFrame(results)

  
    # Determine Best Model per Store/Chain
  
    def get_best_model(group):
        group = group.sort_values(by=['R2', 'RMSE'], ascending=[False, True])
        group['Best_Model'] = [True] + [False] * (len(group) - 1)
        return group

    final_results = final_results.groupby(['Level', 'Store'], dropna=False).apply(get_best_model).reset_index(drop=True)

  
    # Save Best Model Files

    print("\n Saving best models for future use...")
    best_rows = final_results[final_results['Best_Model']]
    for _, row in best_rows.iterrows():
        model_obj = row.get('Trained_model', None)
        if model_obj is not None:
            save_best_model(model_obj, row['Level'], row['Store'], row['Model'])

  
    # Save Results with Best Model Info
 
    best_models_summary = (
        final_results[final_results['Best_Model']]
        .loc[:, ['Level', 'Store', 'Model', 'R2', 'RMSE']]
        .rename(columns={'Model': 'Best_Model_Name', 'R2': 'Best_R2', 'RMSE': 'Best_RMSE'})
    )

    final_results = final_results.merge(best_models_summary, on=['Level', 'Store'], how='left')

    output_path = Path("results") / "longterm_forecast_results.csv"
    output_path.parent.mkdir(exist_ok=True)
    final_results.to_csv(output_path, index=False)
    print(f"\n Results saved to {output_path}")

    summary_path = Path("results") / "longterm_best_models_summary.csv"
    best_models_summary.to_csv(summary_path, index=False)
    print(f" Best model summary saved to {summary_path}")

    return final_results



if __name__ == "__main__":
    run_full_pipeline(horizon_long=20)
