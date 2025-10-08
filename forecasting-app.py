import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from pathlib import Path

# =========================
# Load Data
# =========================
st.set_page_config(page_title="Sales Forecasting Dashboard", layout="wide")

@st.cache_data

def load_data():

    SCRIPT_DIR= Path().resolve()
    long_path =  SCRIPT_DIR/'results'/'longterm_forecast_results.csv'
    short_path = SCRIPT_DIR/'results'/'shortterm_forecast_results_full.csv'
    cleaned_data  = SCRIPT_DIR/'data'/'processed'/'cleaned_data.csv'
    global_test_path = SCRIPT_DIR/'results'/'global_test_result.csv'
    global_train_path = SCRIPT_DIR/'results'/'global_train_result.csv'
    long_df = pd.read_csv(long_path)
    short_df = pd.read_csv(short_path)
    df_cleaned = pd.read_csv(cleaned_data)
    global_test =pd.read_csv(global_train_path)
    global_train =pd.read_csv(global_test_path)
    return short_df, long_df, df_cleaned,global_test,global_train 

short_df, long_df, df_cleaned,global_test,global_train = load_data()

# =========================
# Dashboard Title
# =========================
# st.set_page_config(page_title="Sales Forecasting Dashboard", layout="wide")
st.title("📊 Sales Forecasting Dashboard")
st.markdown("""
This interactive dashboard visualizes **global** and **store-level** sales forecasts.
You can view short-term & long-term predictions, and compare **two stores** side by side.
""")

# =========================
# Global Forecast Plot
# =========================
st.subheader(" Global Chain Forecasting — Actual vs Predicted")

fig_global = go.Figure()
fig_global.add_trace(go.Scatter(
    x=global_test['Date'], y=global_test['Actual'],
    mode='lines', name='Actual', line=dict(color='blue')
))
fig_global.add_trace(go.Scatter(
    x=global_test['Date'], y=global_test["Predicted"],
    mode='lines', name='Predicted', line=dict(color='orange')
))
fig_global.add_trace(go.Scatter(
    x=global_train['Date'], y=global_train.iloc[:,1],
    mode='lines', name='Train', line=dict(color='green')
))
fig_global.update_layout(
    title="Global Forecasting — Actual vs Predicted",
    xaxis_title="Date",
    yaxis_title="Weekly Sales",
    legend=dict(x=0, y=1),
    hovermode="x unified"
)
st.plotly_chart(fig_global, use_container_width=True)

# =========================
# Store-Level Comparison Section
# =========================
st.subheader("🏪 Store-Level Forecasting & Comparison")

store_list = sorted(short_df["Store"].unique())
col_sel1, col_sel2 = st.columns(2)

store1 = col_sel1.selectbox("Select Store 1", store_list, index=0)
store2 = col_sel2.selectbox("Select Store 2 (for comparison)", store_list, index=1)

def plot_store_forecast(df,df_clean, store, title):
    # df['Store'] = df['Store'].astype(int)
    # df['R2'] = df['R2'].astype(float)
    # df['RMSE'] = df['RMSE'].astype(float)
    # print(df.info())
    # print(df_clean.info())

    df_dates = df[df['Store']==10]
    df_copy = df[df['Store']==store]
    df_copy = df_copy.sort_values(by='RMSE')
    df_clean_copy = df_clean[df_clean['Store']==store]

    actuals = df_clean_copy.groupby('Date')['Weekly_Sales'].sum()
    dates = pd.to_datetime(actuals.index)

    forecast_values = [v for v in df_copy['Forecast'].values if pd.notna(v) and v != '']
    val = forecast_values[1] if len(forecast_values) > 0 else None
   
    val_clean = val.strip("[]")


    forecast_floats = [float(x) for x in val_clean.split()]

    forecast =[]

    forecast.append(forecast_floats)

    lst = df_dates['Actual'].values[0].split("\n")
    
    forecast_dates=[]
    for entry in lst:
        parts = entry.split()
        if len(parts) == 2:
            forecast_dates.append(parts[0])

    forecast_dates = pd.to_datetime(forecast_dates)

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=dates, y=actuals,
        mode='lines', name='Actual', line=dict(color='green')
    ))
    fig.add_trace(go.Scatter(
        x=forecast_dates, y=forecast[0],
        mode='lines', name='Predicted', line=dict(color='orange')
    ))
    fig.update_layout(
        title=f"{title} — Store {store}",
        xaxis_title="Date",
        yaxis_title="Weekly Sales",
        legend=dict(x=0, y=1),
        hovermode="x unified"
    )
    return fig

# --- Metrics display ---
def display_metrics(store_num, metrics_df, df):
    dft = df[df['Store']==store_num]
    rows = metrics_df[metrics_df["Store"] == store_num]
    # row = rows.loc[rows["RMSE"].idxmin()]
    # val = best_row["Forecast"]
    row = rows.sort_values(by='RMSE', ascending=True)
    
    type_val = dft['Type'].iloc[0]
    st.metric("Type", f"{type_val}")


    if not row.empty:
       
        r2_val = row["R2"] 
        rmse_vals = [v for v in row['RMSE'].values if v != 0]
        rmse_val = rmse_vals[1]
        # st.metric("R²", f"{r2_val:.3f}")
        st.metric("RMSE", f"{rmse_val:,.0f}")



    else:
        st.info("No metrics available for this store.")

# =========================
# Short-Term Forecast Plots
# =========================
st.markdown("### ⏱ Short-Term Forecast Comparison")

col1, col2 = st.columns(2)
with col1:
    st.markdown(f"**Store {store1}**")
    display_metrics(store1, short_df, df_cleaned)
    st.plotly_chart(plot_store_forecast(short_df, df_cleaned, store1, "Short-Term Forecast"), use_container_width=True, key=f"SHORT_{store1}")

with col2:
    st.markdown(f"**Store {store2}**")
    display_metrics(store2,short_df, df_cleaned)
    st.plotly_chart(plot_store_forecast(short_df,df_cleaned, store2, "Short-Term Forecast"), use_container_width=True, key=f"SHORT_{store2}")

# =========================
# Long-Term Forecast Plots
# =========================
st.markdown("### Long-Term Forecast Comparison")

col3, col4 = st.columns(2)
with col3:
    st.markdown(f"**Store {store1}**")
    display_metrics(store1, long_df, df_cleaned)
    st.plotly_chart(plot_store_forecast(long_df,df_cleaned, store1, "Long-Term Forecast"), use_container_width=True, key=f"LONG_{store1}")

with col4:
    st.markdown(f"**Store {store2}**")
    display_metrics(store2, long_df, df_cleaned)
    st.plotly_chart(plot_store_forecast(long_df,df_cleaned, store2, "Long-Term Forecast"), use_container_width=True,  key=f"LONG_{store2}")
    
# =========================
# Footer
# =========================
st.markdown("---")
st.markdown(" *Tip: Use the dropdown menus above to compare forecasting results between two stores interactively.*")
