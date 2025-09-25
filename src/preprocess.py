
# Importing libraries
from pathlib import Path
import pandas as pd
import numpy as np

# --- Helper Function for Date Extraction 
def extract_from_date(df, date_col):
    
    df['Year'] = df[date_col].dt.year.astype(int)
    df['Month'] = df[date_col].dt.month.astype(int)
    df['WeekOfYear'] = df[date_col].dt.isocalendar().week.astype(int)
    return df 

# --- Main Processing Function ---
def process_data():
    # Defining paths
    SCRIPT_DIR = Path(__file__).resolve().parent
    PROJECT_ROOT = SCRIPT_DIR.parent
    
    # ------------------------------------------------
    # 1. Define all I/O paths (Loading and Saving)
    # ------------------------------------------------
    sales_data_path = PROJECT_ROOT / 'data' / 'raw' / 'sales.csv'
    features_data_path = PROJECT_ROOT / 'data' / 'raw' / 'Features.csv'
    stores_data_path = PROJECT_ROOT / 'data' / 'raw' / 'stores.csv'
    
    # Defining output path
    output_path = PROJECT_ROOT / 'data' / 'processed' / 'preprocessed_data.csv'

    ## 2. Loading Data
    try:
        df_sales = pd.read_csv(sales_data_path)
        df_features = pd.read_csv(features_data_path)
        df_stores = pd.read_csv(stores_data_path)
        print(f"Data loaded successfully!")
    except FileNotFoundError:
        print(f"Error: One or more files were not found. Please check paths.")
        # Exit the function if loading fails
        return 

    # ------------------------------------------------
    # 3. Data Cleaning
    # ------------------------------------------------

    ## Date conversion 
    df_features['Date'] = pd.to_datetime(df_features['Date'], format="%d/%m/%Y")
    df_sales['Date'] = pd.to_datetime(df_sales['Date'], format="%d/%m/%Y")

    ## Handling missing values
    df_features[['MarkDown1','MarkDown2','MarkDown3','MarkDown4','MarkDown5']] = \
        df_features[['MarkDown1','MarkDown2','MarkDown3','MarkDown4','MarkDown5']].fillna(0)

    # Imputing CPI and Unemployment
    df_features['cpi'] = df_features.groupby('Store')['CPI'].transform(lambda x: x.interpolate(method='linear'))
    df_features['unemployment'] = df_features.groupby('Store')['Unemployment'].transform(lambda x: x.interpolate(method='linear'))

    # Dropping old and checking missing values
    df_features.drop(columns=['CPI', 'Unemployment'], inplace=True)
    # df_features.rename(columns={'CPI_new': 'CPI', 'Unemployment_new': 'Unemployment'}, inplace=True)
    print("\nMissing values check in df_features after imputation:")
    print(df_features.isna().sum())

    ## Merging the data
    df_merge1 = pd.merge(df_features, df_stores, on='Store', how='left')
    df_merged = pd.merge(df_sales, df_merge1, on=['Date', 'Store', 'IsHoliday'], how='left')
    print("\nFinal merged dataset info:")
    print(df_merged.info())

    ## Extracting date features & creating total MarkDown
    df_merged = extract_from_date(df_merged.copy(), 'Date')
    df_merged['Total_MarkDown'] = df_merged[['MarkDown1','MarkDown2','MarkDown3','MarkDown4','MarkDown5']].sum(axis=1)

    ## changing Isholiday to int
    df_merged['IsHoliday'] = df_merged['IsHoliday'].astype(int)

    # ------------------------------------------------
    # 4. Exporting the processed data
    # ------------------------------------------------
    
    # make processed data directory if it doesn't exist
    output_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        df_merged.to_csv(output_path, index=False) 
        print(f"\nSuccessfully exported processed data to: {output_path}")

    except Exception as e:
        print(f"\nError exporting data: {e}")
    return

# --- Execution Block ---
if __name__ == "__main__":
    process_data()