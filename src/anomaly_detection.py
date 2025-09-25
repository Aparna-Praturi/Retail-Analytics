# importing libraries
import numpy as np
import pandas as pd
from tqdm import tqdm
from pathlib import Path
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, OneHotEncoder, RobustScaler, MinMaxScaler
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools.tools import add_constant
from statsmodels.tsa.seasonal import STL
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
#from sklearn.svm import OneClassSVM
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

def calculate_vif(df, features):

    X = df[features].copy()

    # Add constant (intercept) for VIF calculation
    X = add_constant(X)

    vif_data = pd.DataFrame()
    vif_data["Feature"] = X.columns
    vif_data["VIF"] = [variance_inflation_factor(X.values, i)
                       for i in range(X.shape[1])]

    # Drop constant row
    vif_data = vif_data[vif_data["Feature"] != "const"]

    return vif_data.reset_index(drop=True)

def feature_engineer(df_merged):
    SCRIPT_DIR = Path(__file__).resolve().parent

    # ------------------------------------------------
    # 1. Define all I/O paths and loading data
    # ------------------------------------------------
    # processed_data_path = SCRIPT_DIR.parent / 'data' / 'processed' / 'preprocessed_data.csv'


    # # Defining output path
    # output_path = SCRIPT_DIR.parent / 'data' / 'processed' / 'cleaned_data.csv'

    # # Loading data
    # try:
    #     df_merged = pd.read_csv(processed_data_path)
    #     print(f"Data loaded successfully!")
    # except FileNotFoundError:
    #     print(f"Error: One or more files were not found. Please check paths.")
    #     return 
    
    # ------------------------------------------------
    # 2. Defining new features
    # ------------------------------------------------
    
    df_merged['store_mean'] = df_merged.groupby('Store')['Weekly_Sales'].transform('mean')
    df_merged['dept_mean'] = df_merged.groupby('Dept')['Weekly_Sales'].transform('mean')
    df_merged['store_dept_mean'] = df_merged.groupby(['Store', 'Dept'])['Weekly_Sales'].transform('mean')

    # Ratio-based features
    df_merged['sales_vs_store'] = df_merged['Weekly_Sales'] / df_merged['store_mean'].dropna()
    df_merged['sales_vs_dept'] = df_merged['Weekly_Sales'] / df_merged['dept_mean'].dropna()

    # # Create boolean flags for MarkDowns
    # markdown_cols = ['MarkDown1', 'MarkDown2', 'MarkDown3', 'MarkDown4', 'MarkDown5']

    # for col in markdown_cols:
    #     df_merged[f'{col}_flag'] = (df_merged[col] > 0).astype(int)
    # ------------------------------------------------
    # 3. Transforming and scaling features
    # ------------------------------------------------

    # 1. log transformation of skewed features
    skewed_cols = ['Weekly_Sales','sales_vs_store', 'MarkDown1', 'MarkDown2', 'MarkDown3',
               'MarkDown4', 'MarkDown5','Total_MarkDown']
    
    for col in (skewed_cols):
        df_merged[f'{col}_transformed'] = np.log1p(df_merged[col]).where(df_merged[col] > 0, 1)

        # Replace any  infinite values with NaN
        df_merged[f'{col}_transformed'] = df_merged[f'{col}_transformed'].replace([np.inf, -np.inf], np.nan)

        # Drop rows with NaN values in 'log_weekly_sales'
        df_merged.dropna(subset=[f'{col}_transformed'], inplace=True)

        # scaling
        #scaler = StandardScaler()
        scaler = RobustScaler()
        df_merged[f'{col}_scaled'] = scaler.fit_transform(df_merged[[f'{col}_transformed']])

    ## 2. Scaling other numeric features
    unskewed_cols = ['Temperature', 'Fuel_Price', 'cpi', 'unemployment', 'Size', 'sales_vs_dept',
                    'WeekOfYear']
    
    print(df_merged.info())

    # Scaling with standard scaler
    for col in unskewed_cols:
    #scaler = StandardScaler()
        scaler = RobustScaler()
        df_merged[f'{col}_scaled'] = scaler.fit_transform(df_merged[[col]])

    ## 3. Frequency encoding store, dept
    cat_cols = ['Dept','Store']
    for col in cat_cols:
        freq_map = df_merged[col].value_counts().to_dict()
        df_merged[f'{col}_id_freq'] = df_merged[col].map(freq_map)
        df_merged[f'{col}_scaled'] = scaler.fit_transform(df_merged[[f'{col}_id_freq']])

    ## 4. Scaling year and month

    # Scaling year by subtracting base year
    base_year = 2010
    df_merged['year_scaled'] = df_merged['Year']-base_year

    # cyclical encoding of month
    df_merged['month_sin_scaled'] = np.sin(2 * np.pi * df_merged['Month'] / 12)
    df_merged['month_cos_scaled'] = np.cos(2 * np.pi * df_merged['Month'] / 12)
    df_merged['weekofyear_scaled'] = np.cos(2 * np.pi * df_merged['WeekOfYear'] /52)

    return(df_merged)

def detect_anomalies():
    SCRIPT_DIR = Path(__file__).resolve().parent

    # ------------------------------------------------
    # 1. Define all I/O paths and loading data
    # ------------------------------------------------
    processed_data_path = SCRIPT_DIR.parent/'data'/'processed'/'preprocessed_data.csv'
    
    # Defining output path
    output_path = SCRIPT_DIR.parent /'data'/'processed'/'cleaned_data.csv'

    RESULTS_DIR = SCRIPT_DIR.parent /'results' 
    RESULTS_DIR.mkdir(parents=True, exist_ok=True) 

    # Loading data
    try:
        df_merged = pd.read_csv(processed_data_path)
        print(f"Data loaded successfully!")
    except FileNotFoundError:
        print(f"Error: One or more files were not found. Please check paths.")
        return
    
    # 2. Perform feature engineering
    fe_df = feature_engineer(df_merged)
    df_merged = fe_df.copy()

    # 3. Selecting relavant features for anomaly detection using correlation with target variable, weekly sales
    numeric_features = fe_df.select_dtypes(include=np.number).columns
    corr = fe_df[numeric_features].corr()['Weekly_Sales_scaled'].sort_values(ascending=False)
    relevant_corr = corr[(corr.index != 'Weekly_Sales_scaled') & (corr.abs() > 0.1)]

    rel_features = ['sales_vs_store_scaled','Dept_id_freq', 'Store_id_freq', 'Size_scaled',
                'sales_vs_dept_scaled']
    
    vif_result = calculate_vif(df_merged, rel_features)
    print(f'vif_result showing feature independence:{vif_result}')


    # -----------------------------------------------------------------
    # 2. Detect anomaly using Isolation Forest and Local outlier Factor
    # -----------------------------------------------------------------
    print('Detecting anomalies using Isolation Forest and Local outlier Factor............')
    # Apply Isolation Forest and Local Outlier Factor
    isolation_forest = IsolationForest(contamination=0.002)
    lof = LocalOutlierFactor(n_neighbors=40, contamination=0.005, novelty=True)

    # Fitting
    if_model = isolation_forest.fit(df_merged[rel_features])
    lof_model = lof.fit(df_merged[rel_features])

    # Predicting
    df_merged['if_anomaly'] = if_model.predict(df_merged[rel_features])
    df_merged['lof_anomaly'] = lof_model.predict(df_merged[rel_features])

    ## Changing anomaly flags from 1/-1 to 0/1

    df_merged[['lof_anomaly','if_anomaly']] = (df_merged[['lof_anomaly','if_anomaly']].replace({1:0, -1:1}))

    # Plotting
    fig,axs = plt.subplots(2, 3, figsize=(8, 5))


    sns.scatterplot(data=df_merged, x='Size', y='Weekly_Sales', hue='if_anomaly',
                    palette={0: 'blue', 1: 'red'}, ax=axs[0,0])

    sns.scatterplot(data=df_merged, x='Total_MarkDown', y='Weekly_Sales', hue='if_anomaly',
                    palette={0: 'blue', 1: 'red'}, ax=axs[0,1])

    sns.scatterplot(data=df_merged, x='sales_vs_store_scaled', y='Weekly_Sales', hue='if_anomaly',
                    palette={0: 'blue', 1: 'red'}, ax=axs[0,2])

    sns.scatterplot(data=df_merged, x='Size', y='Weekly_Sales', hue='lof_anomaly',
                    palette={0: 'blue', 1: 'red'}, ax=axs[1,0])

    sns.scatterplot(data=df_merged, x='Total_MarkDown', y='Weekly_Sales', hue='lof_anomaly',
                    palette={0: 'blue', 1: 'red'}, ax=axs[1,1])

    sns.scatterplot(data=df_merged, x='sales_vs_store_scaled', y='Weekly_Sales', hue='lof_anomaly',
                    palette={0: 'blue', 1: 'red'}, ax=axs[1,2])
    plt.tight_layout()

   
    PLOT_FILENAME = 'anomaly_detection_IF_LOF.png'
    PLOT_SAVE_PATH = RESULTS_DIR / PLOT_FILENAME 
    plt.savefig(PLOT_SAVE_PATH)
    # --- Display the figure (Optional) ---
    #plt.show()

    print(f"Plot saved successfully to: {PLOT_SAVE_PATH}")

    # -----------------------------------------------------------------
    # 3. Detect anomaly using Rolling statistics
    # -----------------------------------------------------------------
    print('Detecting anomalies using Rollinf statistis.............')
    # Sorting by Store, Dept and Date
    df_merged = df_merged.sort_values(by=['Store', 'Dept', 'Date'])

    # Create new columns for anomalies and z-score
    df_merged['rolling_anomaly'] = 0
    df_merged['rolling_zscore'] = np.nan

    # Rolling window size
    window = 8

    # Iterate through each store
    for store in tqdm(df_merged['Store'].unique(), desc='Stores'):
        store_df = df_merged[df_merged['Store'] == store]

        # Iterate through each department in that store
        for dept in store_df['Dept'].unique():
            try:
                group = store_df[store_df['Dept'] == dept]
                group = group.set_index('Date').sort_index()

                # Rolling statistics
                rolling_mean = group['Weekly_Sales'].rolling(window=window, min_periods=1).mean()
                rolling_std = group['Weekly_Sales'].rolling(window=window, min_periods=1).std()

                # Z-score calculation
                z_score = (group['Weekly_Sales'] - rolling_mean) / (rolling_std + 1e-5)
                anomaly_mask = z_score.abs() > 2.0

                # Store results
                idx = df_merged[(df_merged['Store'] == store) & (df_merged['Dept'] == dept)].sort_values('Date').index

            # Assign using these indices
                df_merged.loc[idx, 'rolling_zscore'] = z_score.values
                df_merged.loc[idx, 'rolling_anomaly'] = anomaly_mask.astype(int).values

            except Exception as e:
                print(f"Failed for Store {store}, Dept {dept}: {e}")

    # -----------------------------------------------------------------
    # 4. Detect anomaly using STK Decomposition
    # -----------------------------------------------------------------
    print('Detecting anomalies using STL Decomposition.............')
    # Initialize new columns
    df_merged['stk_trend'] = np.nan
    df_merged['stk_season'] = np.nan
    df_merged['stk_residuals'] = np.nan
    df_merged['stk_anomaly'] = 0

    # Unique store-dept combinations
    combinations = df_merged[['Store', 'Dept']].drop_duplicates()

    for _, row in tqdm(combinations.iterrows()):
        store, dept = row['Store'], row['Dept']
        try:
            mask = (df_merged['Store'] == store) & (df_merged['Dept'] == dept)
            sub_df = df_merged[mask].copy()

            # Time series
            ts = sub_df.set_index('Date')['Weekly_Sales'].sort_index()

            # STL decomposition
            stl = STL(ts, period=52, seasonal=13)  # seasonal usually < period
            result = stl.fit()
            resid, trend, season = result.resid, result.trend, result.seasonal

            # Anomalies (Z-score method)
            z_scores = (resid - resid.mean()) / resid.std()
            anomalies = (z_scores.abs() > 2.5).astype(int)

            # Pack into DataFrame for alignment
            comp_df = pd.DataFrame({
                'Date': ts.index,
                'stk_trend': trend.values,
                'stk_season': season.values,
                'stk_residuals': resid.values,
                'stk_anomaly': anomalies.values
            })

            # Merge back
            df_merged.loc[mask, ['stk_trend','stk_season','stk_residuals','stk_anomaly']] = comp_df[
                ['stk_trend','stk_season','stk_residuals','stk_anomaly']
            ].values

            print(f'STL Decomposition successful for store: {store}, Dept: {dept}')

        except Exception as e:
            print(f"Failed for Store {store}, Dept {dept}: {e}")


    ## Plotting the Seasonality, Trend, residuals and Anomaliesof STL Decomposition

    store_id = 40
    dept_id = 60

    sample = df_merged[(df_merged['Store'] == store_id) & (df_merged['Dept'] == dept_id)].set_index('Date')

    # Plotting
    fig, axs = plt.subplots(2,1, figsize=(12, 12))
    axs=axs.flatten()

    axs[0].plot(sample.index, sample['Weekly_Sales'], label='Weekly Sales')
    axs[0].plot(sample.index, sample['stk_trend'], label='trend', color = 'green')
    axs[0].plot(sample.index, sample['stk_season'], label='seasonality', color = 'orange')
    axs[0].plot(sample.index, sample['stk_residuals'], label='residuals', color = 'red')
    axs[0].legend()
    axs[0].grid(True)


    axs[1].plot(sample.index, sample['Weekly_Sales'], label='Weekly Sales')
    axs[1].scatter(sample[sample['rolling_anomaly'] == 1].index,
                sample[sample['rolling_anomaly'] == 1]['Weekly_Sales'],
                color='red', label='rolling_Anomalies')
    axs[1].scatter(sample[sample['stk_anomaly'] == 1].index,
                sample[sample['stk_anomaly'] == 1]['Weekly_Sales'],
                color='green', label=' stk_Anomalies')
    axs[1].set_title("Weekly Sales with Anomalies")
    axs[1].legend()
    axs[1].grid(True)

    plt.tight_layout()

    PLOT_FILENAME = 'STK Decomposition along with rolling anomalies of Store 40, Dept 60.png'
    PLOT_SAVE_PATH = RESULTS_DIR / PLOT_FILENAME 
    plt.savefig(PLOT_SAVE_PATH)
    print(f"Plot saved successfully to: {PLOT_SAVE_PATH}")

    # Display the figure (Optional)
    #plt.show()

    # Print percentage anomalies
    percent_if_anomalies = (df_merged[df_merged['if_anomaly']==1].shape[0]/df_merged.shape[0])*100
    percent_lof_anomalies = (df_merged[df_merged['lof_anomaly']==1].shape[0]/df_merged.shape[0])*100
    percent_rolling_anomalies = (df_merged[df_merged['rolling_anomaly']==1].shape[0]/df_merged.shape[0])*100
    percent_stk_anomalies = (df_merged[df_merged['stk_anomaly']==1].shape[0]/df_merged.shape[0])*100

    print(f'percent_IF_anomalies: {percent_if_anomalies}')
    print(f'percent_LOF_anomalies: {percent_lof_anomalies}')
    print(f'percent_rolling_anomalies: {percent_rolling_anomalies}')
    print(f'percent_STK_anomalies: {percent_stk_anomalies}')

    # ------------------------------------------------
    # 4. Handling anomalies
    # ------------------------------------------------

    # Seasonal medians by week number
    seasonal_medians = df_merged.groupby('WeekOfYear')['Weekly_Sales'].median()

    # Combine anomaly flags (STL + Rolling)
    df_merged['final_anomaly'] = ((df_merged['stk_anomaly'] == 1) &
    (df_merged['rolling_anomaly'] == 1)).astype(int)

    # Create cleaned version of Weekly_Sales
    df_merged['Weekly_Sales_cleaned'] = df_merged['Weekly_Sales'].copy()

    # Replace only consensus anomalies with seasonal median
    df_merged.loc[df_merged['final_anomaly'] == 1, 'Weekly_Sales_cleaned'] = (
    df_merged.loc[df_merged['final_anomaly'] == 1, 'WeekOfYear'].map(seasonal_medians))

    print(df_merged.info())

    # ------------------------------------------------
    # 4. Exporting the cleaned data
    # ------------------------------------------------
    
    # make cleaned data directory if it doesn't exist
    output_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        df_merged.to_csv(output_path, index=False) 
        print(f"\nSuccessfully exported processed data to: {output_path}")

    except Exception as e:
        print(f"\nError exporting data: {e}")

        
if __name__ == "__main__":
    detect_anomalies() 