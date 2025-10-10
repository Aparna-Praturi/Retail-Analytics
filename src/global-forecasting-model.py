import pandas as pd
import numpy as np
from pathlib import Path
import optuna
from optuna.samplers import TPESampler
from xgboost import XGBRegressor
from sklearn.model_selection import TimeSeriesSplit

from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import joblib
import warnings

warnings.filterwarnings("ignore", category=UserWarning)


# ====================================================
# FEATURE ENGINEERING
# ====================================================

def aggregate_data(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate raw data at Store-Dept-Date level."""
    df = df.groupby(["Store", "Dept", "Date"]).agg({
        "Weekly_Sales_cleaned": "sum",
        "IsHoliday": "mean",
        "Fuel_Price": "mean",
        "cpi": "mean",
        "unemployment": "mean",
        "Size": "mean",
        # "Total_MarkDown": "sum"
    }).reset_index()

    df = df.sort_values(["Store", "Dept", "Date"])
    df["weekofyear"] = df["Date"].dt.isocalendar().week.astype(int)
    df["month"] = df["Date"].dt.month
    df["year"] = df["Date"].dt.year
    return df


def create_lag_rolling_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create lag and rolling features in a chronological, leakage-free way."""
    df = df.sort_values(["Store", "Dept", "Date"]).copy()
    df = df.set_index("Date")

    # Lag features
    for lag in [1, 2]:
        df[f"lag_{lag}"] = (
            df.groupby(["Store", "Dept"])["Weekly_Sales_cleaned"].shift(lag)
        )

    # Rolling means (shifted first to avoid leakage)
    for window in [4, 12]:
        df[f"rolling_{window}"] = (
            df.groupby(["Store", "Dept"])["Weekly_Sales_cleaned"]
            .shift(1)
            .rolling(window, min_periods=1)
            .mean()
        )

    df = df.reset_index()

    # Encode categorical features
    df["Store"] = df["Store"].astype("category").cat.codes
    df["Dept"] = df["Dept"].astype("category").cat.codes

    return df

def rolling_origin_splits(df, n_folds=4, horizon_weeks=12, start_buffer_weeks=26):
    """
    Yields (train_idx, val_idx) where each val block is `horizon_weeks` long.
    start_buffer_weeks: ensure the first train window has enough history.
    """
    df = df.sort_values("Date").reset_index(drop=True)
    dates = df["Date"]
    start_date = dates.min() + pd.Timedelta(weeks=start_buffer_weeks)

    fold_cut_dates = pd.date_range(start=start_date,
                                   end=dates.max() - pd.Timedelta(weeks=horizon_weeks),
                                   periods=n_folds)

    for cut in fold_cut_dates:
        train_idx = df.index[dates <= cut]
        val_mask = (dates > cut) & (dates <= cut + pd.Timedelta(weeks=horizon_weeks))
        val_idx = df.index[val_mask]
        if len(val_idx) > 0:
            yield train_idx, val_idx



# ====================================================
# OPTUNA OBJECTIVE WITH TIMESERIES SPLIT
# ====================================================

def optuna_objective(trial, df, feature_cols, target_col):
    """Optuna objective: uses TimeSeriesSplit CV to evaluate hyperparameters robustly."""
    tscv = TimeSeriesSplit(n_splits=4)
    scores = []

    # Hyperparameter search space
    params = {
        "objective": "reg:squarederror",
        "n_estimators": trial.suggest_int("n_estimators", 100, 800),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "max_depth": trial.suggest_int("max_depth", 3, 12),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "min_child_weight": trial.suggest_float("min_child_weight", 1, 10),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
        "random_state": 42,
        "n_jobs": -1
    }

    for train_idx, val_idx in rolling_origin_splits(df, n_folds=4, horizon_weeks=12, start_buffer_weeks=26):
        train, val = df.iloc[train_idx], df.iloc[val_idx]

        X_train, y_train = train[feature_cols], train[target_col]
        X_val, y_val = val[feature_cols], val[target_col]

       
        model = XGBRegressor(**params)
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            # early_stopping_rounds=30,
            verbose=False
        )

        y_pred = model.predict(X_val)
        scores.append(r2_score(y_val, y_pred))

    return np.mean(scores)


# ====================================================
# 3️TRAIN FINAL MODEL & EVALUATE
# ====================================================

def train_and_evaluate(df, best_params, feature_cols, target_col, horizon=20):
   
    last_train_date = df["Date"].max() - pd.Timedelta(weeks=horizon)

    train = df[df["Date"] <= last_train_date].dropna()
    test = df[df["Date"] > last_train_date].dropna()


    X_train, y_train = train[feature_cols], train[target_col]
    X_test, y_test = test[feature_cols], test[target_col]

    model = XGBRegressor(**best_params)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    print("\ Final Evaluation")
    print(f"RMSE: {rmse:.2f}")
    print(f"R²:   {r2:.3f}")

    # naive: use lag_h as the prediction
    h = 12  # match your horizon
    test = test.copy()
    test['y_baseline'] = test[target_col].shift(h)
   
    mask = test["y_baseline"].notna()
    baseline_r2  = r2_score(test.loc[mask,"Weekly_Sales_cleaned"], test.loc[mask,"y_baseline"])
    baseline_rmse = np.sqrt(mean_squared_error(test.loc[mask,"Weekly_Sales_cleaned"], 
                                               test.loc[mask,"y_baseline"]))
                                   
    print(f"Naive lag-{h} — RMSE:  R²: {baseline_r2:.3f}")


    return model, test["Date"], y_test, y_pred, rmse, r2, X_train, y_train


# ====================================================
# PLOTTING & SAVING
# ====================================================

def plot_predictions(dates, df, y_test, y_pred, save_path: Path, chain_test_result_path,chain_train_result_path):
    last_train_date = df["Date"].max() - pd.Timedelta(weeks=20)
    train_df = df[df["Date"] <= last_train_date].dropna()
    

    train_grouped = train_df.groupby('Date').agg({'Weekly_Sales_cleaned':sum})

    df_result = pd.DataFrame({
        'Date': dates,
        'Actual': y_test,
        'Predicted': y_pred,   
    })
    grouped = df_result.groupby('Date').agg({'Actual':sum, 'Predicted': sum})
    
    grouped.to_csv(chain_test_result_path, index=True)
    train_grouped.to_csv(chain_train_result_path, index=True)
    



    plt.figure(figsize=(14, 6))
    plt.plot(grouped.index, grouped['Actual'], label='Actual', color='red')
    plt.plot(grouped.index, grouped['Predicted'], label='Predicted', color='green')
    plt.plot(train_grouped.index, train_grouped['Weekly_Sales_cleaned'], label='Train', color='black')
    plt.title('Global XGBoost Forecasting — Actual vs Predicted')
    plt.xlabel('Date')
    plt.ylabel('Weekly Sales')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.show()

    print(f" Plot saved to: {save_path}")


def save_model_and_metrics(model, params, rmse, r2, models_dir: Path, results_dir: Path):
    models_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    # Save model
    model_path = models_dir / "xgboost_global_model.pkl"
    joblib.dump(model, model_path)

    # Save metrics
    metrics_path = results_dir / "global_model_metrics.csv"
    metrics_df = pd.DataFrame([{"RMSE": rmse, "R2": r2, **params}])
    metrics_df.to_csv(metrics_path, index=False)

    print(f" Model saved to: {model_path}")
    print(f" Metrics saved to: {metrics_path}")

 


# ====================================================
#  MAIN PIPELINE
# ====================================================

def run_pipeline(n_trials=30):
    SCRIPT_DIR = Path(__file__).resolve().parent
    data_path = SCRIPT_DIR.parent / 'data' / 'processed' / 'cleaned_data.csv'
    results_dir = SCRIPT_DIR.parent / 'results'
    models_dir = SCRIPT_DIR.parent / 'saved_models'

    print(" Loading data...")
    df = pd.read_csv(data_path, parse_dates=["Date"])
    print(f" Data loaded: {df.shape}")

    # Feature Engineering
    print(" Feature engineering...")
    df = aggregate_data(df)
    df = create_lag_rolling_features(df)

    target_col = "Weekly_Sales_cleaned"
    feature_cols = [
        "Store", "Dept", "IsHoliday", "Fuel_Price", "cpi", "unemployment",
        "Size",  "weekofyear", "month", "year"
        # "lag_1", "lag_2",
        #   "rolling_4", "rolling_12"
    ]

    # Optuna Hyperparameter Tuning
    print(f"\n Starting Optuna hyperparameter tuning ({n_trials} trials)...")
    sampler = TPESampler(seed=42)
    study = optuna.create_study(direction="maximize", sampler=sampler)
    study.optimize(
        lambda trial: optuna_objective(trial, df, feature_cols, target_col),
        n_trials=n_trials,
        show_progress_bar=True
    )

    print("\n Best Hyperparameters:")
    for k, v in study.best_params.items():
        print(f"  {k}: {v}")

    # Train Final Model & Evaluate
    model, dates, y_test, y_pred, rmse, r2, X_train, y_train = train_and_evaluate(
        df, study.best_params, feature_cols, target_col
    )

    # Plot
    plot_path = results_dir / "xgboost_predictions.png"
    chain_test_result_path = results_dir / "Global_test_result.csv"
    chain_train_result_path = results_dir / "Global_train_result.csv"
    plot_predictions(dates,df, y_test, y_pred, plot_path, chain_test_result_path,chain_train_result_path)

    # Save model and metrics
    save_model_and_metrics(model, study.best_params, rmse, r2, models_dir, results_dir)


# ====================================================
# ENTRY POINT
# ====================================================

if __name__ == "__main__":
    run_pipeline(n_trials=30)

