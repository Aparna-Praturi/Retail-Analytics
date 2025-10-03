import streamlit as st
import pandas as pd
import joblib
from pathlib import Path
import matplotlib.pyplot as plt

# ================================
# 📌 Config Paths
# ================================
CLEAN_DATA = Path("data/processed/cleaned_data.csv")
SHORTTERM_CSV = Path("results/shortterm_best_models.csv")
LONGTERM_CSV = Path("results/longterm_best_models.csv")
CHAIN_MODEL_CSV = Path("results/chain_best_model.csv")
CHAIN_MODEL_PATH = Path("saved_models/Chain/Best_Chain_Model.pkl")
MODELS_DIR = Path("saved_models/Store")

# ================================
# 🧠 Load Data and Model Helpers
# ================================
@st.cache_data
def load_data():
    df = pd.read_csv(CLEAN_DATA, parse_dates=["Date"])
    short_df = pd.read_csv(SHORTTERM_CSV) if SHORTTERM_CSV.exists() else pd.DataFrame()
    long_df = pd.read_csv(LONGTERM_CSV) if LONGTERM_CSV.exists() else pd.DataFrame()
    return df, short_df, long_df

@st.cache_data
def load_chain_model():
    perf_df = pd.read_csv(CHAIN_MODEL_CSV) if CHAIN_MODEL_CSV.exists() else None
    model = joblib.load(CHAIN_MODEL_PATH) if CHAIN_MODEL_PATH.exists() else None
    return model, perf_df

def load_model(store: int, model_name: str):
    filename = f"{model_name}_Store_{store}.pkl"
    filepath = MODELS_DIR / filename
    return joblib.load(filepath) if filepath.exists() else None

def plot_forecast(actual, forecast, title):
    fig, ax = plt.subplots(figsize=(10, 4))
    actual.plot(ax=ax, label="Actual", color="black")
    forecast.plot(ax=ax, label="Forecast", color="tomato")
    ax.set_title(title)
    ax.legend()
    return fig

# ================================
# 🌐 Streamlit App Layout
# ================================
st.set_page_config(page_title="Retail Forecasting Dashboard", layout="centered")
st.title("🛍️ Retail Demand Forecasting Dashboard")

st.markdown("""
This dashboard visualizes **Chain-level forecasts** and allows you to explore **Store-level forecasts** 
for both **short-term** (8 weeks) and **long-term** (52 weeks), using the best models selected during training.
""")

# ================================
# 📈 Chain-level Forecast
# ================================
st.subheader("📌 Chain-Level Forecast")

df, shortterm_results, longterm_results = load_data()
chain_model, chain_perf = load_chain_model()

if chain_model is None or chain_perf is None or chain_perf.empty:
    st.warning("⚠️ Chain-level best model not found. Please train and save it first.")
else:
    model_name_chain = chain_perf.iloc[0]["Model"]
    r2_chain = chain_perf.iloc[0]["R2"]
    rmse_chain = chain_perf.iloc[0]["RMSE"]

    st.write(f"**Best Model:** {model_name_chain}")
    st.write(f"**R²:** {r2_chain:.3f}")
    st.write(f"**RMSE:** {rmse_chain:,.0f}")

    chain_series = (
        df.groupby("Date")["Weekly_Sales"]
        .sum()
        .asfreq("W-FRI")
        .fillna(0)
        .sort_index()
    )

    horizon = 52
    future_index = pd.date_range(chain_series.index[-1], periods=horizon+1, freq="W-FRI")[1:]

    # Forecast based on model type
    if model_name_chain.lower() == "sarima":
        forecast_chain = chain_model.get_forecast(steps=horizon).predicted_mean
        forecast_chain.index = future_index

    elif model_name_chain.lower() == "xgboost":
        # TODO: implement XGBoost chain forecast
        forecast_chain = pd.Series([chain_series.iloc[-1]] * horizon, index=future_index)

    elif model_name_chain.lower() == "prophet":
        # TODO: implement Prophet chain forecast
        forecast_chain = pd.Series([chain_series.iloc[-1]] * horizon, index=future_index)

    else:
        forecast_chain = pd.Series([chain_series.iloc[-1]] * horizon, index=future_index)

    st.pyplot(plot_forecast(chain_series, forecast_chain, f"Chain-Level Forecast ({model_name_chain})"))

    with st.expander("🔍 View Model Details"):
        st.write(chain_perf)

# ================================
# 🏪 Store-level Forecast Section
# ================================
st.subheader("🏪 Store-Level Forecast Explorer")

store_input = st.number_input("Enter Store Number:", min_value=1, max_value=int(df["Store"].max()), step=1)

if st.button("Show Store Forecast"):
    store = int(store_input)
    store_series = (
        df[df['Store'] == store]
        .groupby('Date')['Weekly_Sales']
        .sum()
        .asfreq('W-FRI')
        .fillna(0)
        .sort_index()
    )

    # --- Short-Term Forecast ---
    st.markdown("### ⏱ Short-Term Forecast (8 Weeks)")
    short_row = shortterm_results[shortterm_results["Store"] == store]
    if short_row.empty:
        st.warning(f"No short-term model found for Store {store}")
    else:
        model_name_short = short_row.iloc[0]["Model"]
        r2_short = short_row.iloc[0]["R2"]
        rmse_short = short_row.iloc[0]["RMSE"]
        st.write(f"**Best Model:** {model_name_short}")
        st.write(f"**R²:** {r2_short:.3f}")
        st.write(f"**RMSE:** {rmse_short:,.0f}")

        horizon_short = 8
        future_index_short = pd.date_range(store_series.index[-1], periods=horizon_short+1, freq="W-FRI")[1:]

        if model_name_short == "SARIMA":
            model = load_model(store, "SARIMA")
            if model:
                forecast = model.get_forecast(steps=horizon_short).predicted_mean
                forecast.index = future_index_short
                st.pyplot(plot_forecast(store_series, forecast, f"Store {store} - Short-Term SARIMA Forecast"))

        elif model_name_short == "Hybrid":
            sarima_model = load_model(store, "SARIMA")
            lgb_model = load_model(store, "Hybrid_LGBM")
            if sarima_model and lgb_model:
                sarima_fc = sarima_model.get_forecast(steps=horizon_short).predicted_mean
                sarima_fc.index = future_index_short
                # For simplicity: using SARIMA forecast only here, hybrid residual can be added later
                forecast = sarima_fc
                st.pyplot(plot_forecast(store_series, forecast, f"Store {store} - Short-Term Hybrid Forecast"))

        elif model_name_short == "Naive":
            forecast = pd.Series([store_series.iloc[-1]] * horizon_short, index=future_index_short)
            st.pyplot(plot_forecast(store_series, forecast, f"Store {store} - Short-Term Naive Forecast"))

        with st.expander("🔍 View Short-Term Model Details"):
            st.write(short_row)

    # --- Long-Term Forecast ---
    st.markdown("### 📅 Long-Term Forecast (52 Weeks)")
    long_row = longterm_results[longterm_results["Store"] == store]
    if long_row.empty:
        st.warning(f"No long-term model found for Store {store}")
    else:
        model_name_long = long_row.iloc[0]["Model"]
        r2_long = long_row.iloc[0]["R2"]
        rmse_long = long_row.iloc[0]["RMSE"]
        st.write(f"**Best Model:** {model_name_long}")
        st.write(f"**R²:** {r2_long:.3f}")
        st.write(f"**RMSE:** {rmse_long:,.0f}")

        # Placeholder for long-term forecast
        horizon_long = 52
        future_index_long = pd.date_range(store_series.index[-1], periods=horizon_long+1, freq="W-FRI")[1:]
        forecast_long = pd.Series([store_series.iloc[-1]] * horizon_long, index=future_index_long)

        st.pyplot(plot_forecast(store_series, forecast_long, f"Store {store} - Long-Term Forecast"))
        with st.expander("🔍 View Long-Term Model Details"):
            st.write(long_row)
