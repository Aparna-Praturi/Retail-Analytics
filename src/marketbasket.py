import pandas as pd
import numpy as np
import seaborn as sns
from pathlib import Path
import matplotlib.pyplot as plt


## Helper function to draw correlation heatmap
def correlation_heatmap(df_merged):

    # Defining results path
    SCRIPT_DIR = Path(__file__).resolve().parent
    RESULTS_DIR = SCRIPT_DIR.parent /'results' 
    RESULTS_DIR.mkdir(parents=True, exist_ok=True) 

    
    ## 1.  Developing a correlation matrix
    dept_sales_pivot = df_merged.groupby(['Date', 'Dept'])['Weekly_Sales_cleaned'].sum().unstack(fill_value=0)
    correlation_matrix = dept_sales_pivot.corr()

    ## 2.  Finding strongly positvely and strongly negatively correlated departments

    # Rename index and columns
    correlation_matrix.index.name = 'Dept1'
    correlation_matrix.columns.name = 'Dept2'

    # stack and reset index
    corr_pairs = correlation_matrix.stack().reset_index()
    corr_pairs.columns = ['Dept1', 'Dept2', 'Correlation']


    #  keep only strong positive correlations
    strong_positive = corr_pairs[
        (corr_pairs['Dept1'] != corr_pairs['Dept2']) &
        (corr_pairs['Correlation'] > 0.85)
    ]

    strong_positive['pair'] = strong_positive.apply(
        lambda x: tuple(sorted([x['Dept1'], x['Dept2']])), axis=1
    )

    # Drop duplicates based on the sorted pair
    strong_positive = strong_positive.drop_duplicates(subset='pair')


    # strong negative correlations
    strong_negative = corr_pairs[
        (corr_pairs['Dept1'] != corr_pairs['Dept2']) &
        (corr_pairs['Correlation'] < -0.65)
    ]

    strong_negative['pair'] = strong_negative.apply(
        lambda x: tuple(sorted([x['Dept1'], x['Dept2']])), axis=1
    )

    strong_negative = strong_negative.drop_duplicates(subset='pair').drop(columns='pair')

    ## 3. Creating a pivot table for strongly correlated pairs
    pivot_df = strong_positive.pivot(index='Dept1', columns='Dept2', values='Correlation')
    pivot_df = pivot_df.fillna(0)  # make symmetric if needed

    sns.heatmap(pivot_df, annot=True, cmap='YlGnBu')
    plt.title('Strong Positive Correlations')
    PLOT_FILENAME = 'Departments with strong positive correlations'
    PLOT_SAVE_PATH = RESULTS_DIR / PLOT_FILENAME 
    plt.savefig(PLOT_SAVE_PATH)
    plt.show()

    return 

## helper function for market basket analysis
def marketbasket(df_merged):

    # making a df with only the required features
    df_mb = df_merged[['Store', 'Dept', "Date", "Weekly_Sales_cleaned"]]

    
    # Create a pivot table: Rows = Date (or Store-Date), Columns = Depts, Values = Presence (1/0)
    basket = df_mb[df_mb.loc[:,'Weekly_Sales_cleaned'] > 0].copy()
    basket['Present'] = 1

    # Group by Store-Date and pivot
    basket_matrix = basket.pivot_table(index=['Store', 'Date'],
                                    columns='Dept',
                                    values='Present',
                                    fill_value=0)
    
    # Choosing the departments which are frequent
    frequent_depts = basket_matrix.columns[(basket_matrix.sum(axis=0) / len(basket_matrix)) >= 0.15]
    basket_matrix = basket_matrix[frequent_depts]

    basket_matrix = basket_matrix.astype(bool)


    frequent_itemsets = fpgrowth(basket_matrix, min_support=0.3,max_len = 3, use_colnames=True)


    rules = association_rules(frequent_itemsets, metric='lift', min_threshold=1.0)
    strong_rules = rules[(rules['confidence'] > 0.85) & (rules['lift'] > 1.45)]


