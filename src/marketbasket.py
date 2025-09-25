import pandas as pd
import numpy as np
import seaborn as sns
from pathlib import Path
import time
import matplotlib.pyplot as plt
import networkx as nx
from mlxtend.frequent_patterns import fpgrowth
from mlxtend.frequent_patterns import apriori, association_rules


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
    ].copy()

    strong_positive['pair'] = strong_positive.apply(
        lambda x: tuple(sorted([x['Dept1'], x['Dept2']])), axis=1
    )

    # Drop duplicates based on the sorted pair
    strong_positive = strong_positive.drop_duplicates(subset='pair')


    # strong negative correlations
    strong_negative = corr_pairs[
        (corr_pairs['Dept1'] != corr_pairs['Dept2']) &
        (corr_pairs['Correlation'] < -0.65)
    ].copy()

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
def marketbasket():

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
    
    correlation_heatmap(df_merged)

 
    # making a df with only the required features
    df_mb = df_merged[['Store', 'Dept', "Date", "Weekly_Sales_cleaned"]]

    print("creating a pivot table for baskets.......")
    # Create a pivot table: Rows = Date (or Store-Date), Columns = Depts, Values = Presence (1/0)
    basket = df_mb[df_mb.loc[:,'Weekly_Sales_cleaned'] > 0].copy()
    basket['Present'] = 1

    # Group by Store-Date and pivot
    basket_matrix = basket.pivot_table(index=['Store', 'Date'],
                                    columns='Dept',
                                    values='Present',
                                    fill_value=0)
    
    # Choosing the departments which are frequent
    frequent_depts = basket_matrix.columns[(basket_matrix.sum(axis=0) / len(basket_matrix)) >= 0.25]
    basket_matrix = basket_matrix[frequent_depts]

    basket_matrix = basket_matrix.astype(bool)

    print(basket_matrix.shape)

    print("Choosing frequent itsemsets..........")
    start = time.time()
    frequent_itemsets = apriori(basket_matrix, min_support=0.25,max_len = 3, use_colnames=True)
    print(f"Done! Found {len(frequent_itemsets)} itemsets in {time.time()-start:.2f} seconds")

    print("calculating association rules...........")
    rules = association_rules(frequent_itemsets, metric='lift', min_threshold=1.0).dropna()
    
    # Filtering out strong rules
    strong_rules = rules[(rules['confidence'] > 0.7) & (rules['lift'] > 1.2)].copy()
    # strong_rules = rules[(rules['confidence'] > 0.85) & (rules['lift'] > 1.4) &(rules['support'] > 0.3)].copy()
    

    print("Visualising strong association rules..............")
    plt.figure(figsize=(8,6))
    plt.scatter(strong_rules['support'], strong_rules['confidence'],
                alpha=0.7, c=strong_rules['lift'], cmap='viridis', edgecolor='k')
    plt.colorbar(label='Lift')
    plt.xlabel('Support')
    plt.ylabel('Confidence')
    plt.title('Association Rules: Support vs Confidence')
    PLOT_FILENAME = 'Support vs Confidence'
    PLOT_SAVE_PATH = RESULTS_DIR / PLOT_FILENAME 
    plt.savefig(PLOT_SAVE_PATH)
    plt.show()

    print("Visualising lift and confidence..............")

    def fs_to_str(fs):
        return ', '.join(map(str, list(fs)))
    strong_rules['antecedents'] = strong_rules['antecedents'].apply(fs_to_str)
    strong_rules['consequents'] = strong_rules['consequents'].apply(fs_to_str)

    # print top rules
    top_rules = strong_rules.sort_values("lift", ascending=False).head(50)

    # Show table
    print(top_rules[['antecedents', 'consequents', 'support', 'confidence', 'lift']])

    #save table
    csv_path = RESULTS_DIR / "top_rules.csv"
    top_rules.to_csv(csv_path, index=False)

    #one to one rules
    one_to_one_rules = strong_rules[
    strong_rules['antecedents'].apply(lambda x: len(x.split(',')) == 1) &
    strong_rules['consequents'].apply(lambda x: len(x.split(',')) == 1)]
    pivot1 = one_to_one_rules.pivot_table(index='antecedents', columns='consequents', values='lift', fill_value=0)


    pivot = strong_rules.pivot_table(index='antecedents', columns='consequents', values='lift', fill_value=0)

    plt.figure(figsize=(12,12))
    
    sns.heatmap(pivot1, annot=False, cmap="YlGnBu", fmt=".2f")
    plt.title("Lift between Antecedents and Consequents")
    PLOT_FILENAME = 'Lift between 1-1 Antecedents and Consequents'
    PLOT_SAVE_PATH = RESULTS_DIR / PLOT_FILENAME 
    plt.savefig(PLOT_SAVE_PATH)
    plt.show()

    plt.figure(figsize=(12,12))
    
    sns.heatmap(pivot, annot=False, cmap="YlGnBu", fmt=".2f")
    plt.title("Lift between Antecedents and Consequents")
    PLOT_FILENAME = 'Lift between Antecedents and Consequents'
    PLOT_SAVE_PATH = RESULTS_DIR / PLOT_FILENAME 
    plt.savefig(PLOT_SAVE_PATH)
    plt.show()

   

    # Visualize strong rules as a network
    print("Visualising as Network graph..............")
    top_rules = strong_rules.sort_values("lift", ascending=False).head(30)
    
    # Build graph
    G = nx.from_pandas_edgelist(
        top_rules, 
        source='antecedents', 
        target='consequents', 
        edge_attr=['lift','confidence','support']
    )

    # Node size = sum of supports
    node_support = {}
    for _, row in top_rules.iterrows():
        node_support[row['antecedents']] = node_support.get(row['antecedents'], 0) + row['support']
        node_support[row['consequents']] = node_support.get(row['consequents'], 0) + row['support']

    node_sizes = [3000 * node_support[n] for n in G.nodes()]  # scale factor

    # Edge width = confidence, edge color = lift
    edge_widths = [d['confidence']*2 for (_,_,d) in G.edges(data=True)]
    edge_colors = [d['lift'] for (_,_,d) in G.edges(data=True)]

    plt.figure(figsize=(14,10))
    pos = nx.spring_layout(G, k=0.4, seed=42)

    nodes = nx.draw_networkx_nodes(G, pos, node_size=node_sizes, node_color="skyblue", alpha=0.8)
    edges = nx.draw_networkx_edges(G, pos, width=edge_widths, edge_color=edge_colors, edge_cmap=plt.cm.viridis)
    nx.draw_networkx_labels(G, pos, font_size=9)

    plt.title("Department Association Network (Node=Support, Edge=Lift/Confidence)", fontsize=14)
    plt.colorbar(edges, label="Lift")
    plt.axis("off")
    

    PLOT_FILENAME = 'Network plot'
    PLOT_SAVE_PATH = RESULTS_DIR / PLOT_FILENAME 
    plt.savefig(PLOT_SAVE_PATH)
    plt.show()

    return

if __name__ == "__main__":
    marketbasket()
   



