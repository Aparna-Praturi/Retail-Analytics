# import libraries
import pandas as pd
import numpy as np
from statsmodels.tsa.seasonal import STL
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, OneHotEncoder, RobustScaler, MinMaxScaler
from pathlib import Path
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering


## Helper function to extract store features
def find_store_features(df_merged):
   
    stores_df = df_merged.copy()

    # Group data by store
    store_groups = stores_df.groupby('Store')
    features = []

    for store, store_df in store_groups:

        # Aggregate to store-level weekly sales
        weekly_sales = store_df.groupby('Date')['Weekly_Sales_cleaned'].sum().sort_index()

        # Descriptive stats
        total_sales = weekly_sales.sum()
        std_sales = weekly_sales.std()
        max_sales = weekly_sales.max()
        min_sales = weekly_sales.min()
        median_sales = weekly_sales.median()
        mean_sales = weekly_sales.mean()

        # Seasonality strength via STL
        stl = STL(weekly_sales, period=52, robust=True)
        result = stl.fit()
        trend = result.trend
        seasonality = result.seasonal
        resid = result.resid
        seasonality_strength = 1 - (np.var(resid) / np.var(result.observed))
        trend_strength = 1 - (np.var(resid) / np.var(resid + trend))


        #  Volatility via rolling IQR
        rolling_iqr = weekly_sales.rolling(window=12).quantile(0.75) - weekly_sales.rolling(window=12).quantile(0.25)
        volatility = rolling_iqr.mean()

        # Promotions: % weeks with markdowns
        store_markdowns = store_df.groupby('Date')['Total_MarkDown'].sum()
        percent_promo_weeks = (store_markdowns > 0).mean()

        # Markdown impact: correlation with sales
        sales_aligned = weekly_sales.loc[store_markdowns.index]
        corr_markdown_sales = store_markdowns.corr(sales_aligned) if not store_markdowns.isnull().all() else np.nan

        # Anomalies
        rolling_flags = store_df.groupby('Date')['rolling_anomaly'].sum()
        stk_flags = store_df.groupby('Date')['stk_anomaly'].sum()
        anomaly_flags = rolling_flags + stk_flags
        percent_anomalies = anomaly_flags.mean()

        # Sales per store size
        store_size = stores_df.loc[stores_df['Store'] == store, 'Size'].values[0]
        sales_per_size = mean_sales / store_size if store_size != 0 else np.nan
        sales_vs_store = stores_df.loc[stores_df['Store'] == store, 'sales_vs_store'].mean()


        # Append to feature list
        features.append({
            'Store': store,
            'median_Sales': median_sales,
            #'total_Sales': total_sales,
            'Std_Sales': std_sales,
            # 'Max_Sales': max_sales,
            # 'Min_Sales': min_sales,
            'Seasonality_Strength': seasonality_strength,
            'Volatility': volatility,
            #'Percent_Promo_Weeks': percent_promo_weeks,
            'Corr_Markdown_Sales': corr_markdown_sales,
            'Percent_Anomalies': percent_anomalies,
            'Sales_per_Size': sales_per_size,
            #'sales_vs_store': sales_vs_store
        })

    # Final DataFrame
    store_features = pd.DataFrame(features)

    return(store_features)

## Helper function to find optimalnumber of clusters
def find_clusters(store_features):
    print("Finding the optimal number of clusters...............")
    SCRIPT_DIR = Path(__file__).resolve().parent
    
    RESULTS_DIR = SCRIPT_DIR.parent /'results' 
    RESULTS_DIR.mkdir(parents=True, exist_ok=True) 
    ## Using Elbow method to find optimal number of clusters

    # Transform & scale features
    transformed = np.log1p(store_features.drop(columns=['Store']))
    scaler = MinMaxScaler()
    scaled_features = scaler.fit_transform(transformed)

    # Elbow WCSS
    wcss = []
    for k in range(1, 11):
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(scaled_features)
        wcss.append(kmeans.inertia_)

    # Silhouette scores
    sil_scores = []
    k_range_sil = range(2, 11)
    for k in k_range_sil:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(scaled_features)
        sil_scores.append(silhouette_score(scaled_features, labels))

    # Plotting
    fig, axs = plt.subplots(1, 2, figsize=(12, 5))

    # Elbow
    axs[0].plot(range(1, 11), wcss, marker='o')
    axs[0].set_title('Elbow Method')
    axs[0].set_xlabel('Number of Clusters (k)')
    axs[0].set_ylabel('WCSS (Inertia)')
    axs[0].grid(True)

    # Silhouette score
    axs[1].plot(list(k_range_sil), sil_scores, marker='o', color='orange')
    axs[1].set_title('Silhouette Score')
    axs[1].set_xlabel('Number of Clusters (k)')
    axs[1].set_ylabel('Silhouette Score')
    axs[1].grid(True)

    plt.tight_layout()
    PLOT_FILENAME = 'Elbow method and silhouette score for number of clusters'
    PLOT_SAVE_PATH = RESULTS_DIR / PLOT_FILENAME 
    plt.savefig(PLOT_SAVE_PATH)
    plt.show()
    return scaled_features

## Helper function to try out different segmentation methods
def store_segmentation_methods(scaled_features):

    SCRIPT_DIR = Path(__file__).resolve().parent
    RESULTS_DIR = SCRIPT_DIR.parent /'results' 
    RESULTS_DIR.mkdir(parents=True, exist_ok=True) 

    print("Trying various segmentation techniques..............")
        
    ## Segmentation with variopus methods
    X = scaled_features
    pca = PCA(n_components=0.95)
    X_scaled = pca.fit_transform(X)


    #  PCA (for visualization )
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)

    # Clustering and Evaluation
    results = {}

    ## 1. KMeans
    print('Trying Kmeans clustering.........')
    kmeans = KMeans(n_clusters=2, random_state=42)
    labels_kmeans = kmeans.fit_predict(X_scaled)
    score_kmeans = silhouette_score(X_scaled, labels_kmeans)
    results['KMeans'] = score_kmeans

    ## 2. DBSCAN
    print('Trying DBScan clustering.........')
    dbscan = DBSCAN(eps=0.3, min_samples=7)
    labels_dbscan = dbscan.fit_predict(X_scaled)
    # Exclude noise (-1 label) for silhouette
    mask = labels_dbscan != -1
    if len(set(labels_dbscan[mask])) > 1:
        score_dbscan = silhouette_score(X_scaled[mask], labels_dbscan[mask])
    else:
        score_dbscan = -1
    results['DBSCAN'] = score_dbscan

    ## 3. Agglomerative
    print('Trying Agglomerative clustering.........')
    agg = AgglomerativeClustering(n_clusters=2)
    labels_agg = agg.fit_predict(X_scaled)
    score_agg = silhouette_score(X_scaled, labels_agg)
    results['Agglomerative'] = score_agg

    # Printing Results
    print("Silhouette Scores:")
    for method, score in results.items():
        print(f"{method}: {score:.3f}")

    #  Plot Clusters using PCA
    plt.figure(figsize=(14, 4))
    for i, (name, labels) in enumerate(zip(['KMeans', 'DBSCAN', 'Agglomerative'],
                                        [labels_kmeans, labels_dbscan, labels_agg])):
        plt.subplot(1, 3, i+1)
        sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1], hue=labels, palette='Set2', legend=False)
        plt.title(name)
    plt.tight_layout()

    PLOT_FILENAME = 'Clustering using various methods'
    PLOT_SAVE_PATH = RESULTS_DIR / PLOT_FILENAME 
    plt.savefig(PLOT_SAVE_PATH)
    print(" saved results of different segmentation methods to results")
    plt.show()

    return

## Main function for store segmentation
def store_segmentation():
       
    # ------------------------------------------------
    # 1. Define all I/O paths and loading data
    # ------------------------------------------------
    SCRIPT_DIR = Path(__file__).resolve().parent
    clean_data_path = SCRIPT_DIR.parent/'data'/'processed'/'cleaned_data.csv'
    RESULTS_DIR = SCRIPT_DIR.parent /'results' 
    RESULTS_DIR.mkdir(parents=True, exist_ok=True) 

    try:
        df_merged = pd.read_csv(clean_data_path)
        print(f"Data loaded successfully!")
    except FileNotFoundError:
        print(f"Error: One or more files were not found. Please check paths.")
        return

    # Determone store features
    df_store = find_store_features(df_merged)

    # Finding clusters
    scaled_features = find_clusters(df_store)

    # Running segmentation methods
    
    store_segmentation_methods(scaled_features)

    # Enter optimal k using graphs
    optimal_k = 2
    kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=5)

    df_store['Cluster'] = kmeans.fit_predict(scaled_features)

    ## Visualising features for different clusters

    cols = df_store.drop(columns=['Store', 'Cluster']).columns

    fig, axs = plt.subplots(4,2, figsize=(12, 12))
    axs=axs.flatten()
    for i, col in enumerate(cols):
        sns.boxplot(x='Cluster', y= col, data=df_store,ax=axs[i])
        axs[i].set_title(col)
        axs[i].grid(True)
    plt.tight_layout()
    PLOT_FILENAME = 'Features for different clusters'
    PLOT_SAVE_PATH = RESULTS_DIR / PLOT_FILENAME 
    plt.savefig(PLOT_SAVE_PATH)
    print('Saved feature plots of various clusters to results')

    return
if __name__ == "__main__":
    store_segmentation()
   





