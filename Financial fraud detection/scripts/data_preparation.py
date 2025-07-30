import os
import pandas as pd
import networkx as nx
import pickle
from sklearn.preprocessing import StandardScaler

def prepare_data():
    # Define paths for files, looking in the data folder by default
    # First, try to find the dataset in the data/elliptic_bitcoin_dataset folder
    possible_paths = [
        os.path.join(os.getcwd(), 'data', 'elliptic_bitcoin_dataset'),  # Current working directory
        os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data', 'elliptic_bitcoin_dataset'),  # Project root
        'C:/Users/Administrator/Downloads/elliptic_dev_dataset'  # Original path as fallback
    ]
    
    # Find the first path that contains the required files
    base_path = None
    for path in possible_paths:
        if (os.path.exists(os.path.join(path, 'elliptic_txs_classes.csv')) and
            os.path.exists(os.path.join(path, 'elliptic_txs_edgelist.csv')) and
            os.path.exists(os.path.join(path, 'elliptic_txs_features.csv'))):
            base_path = path
            print(f"Found dataset in: {base_path}")
            break
    
    if base_path is None:
        raise FileNotFoundError("Could not find the elliptic_bitcoin_dataset in any of the expected locations. "
                               "Please place the dataset in the data/elliptic_bitcoin_dataset folder.")
    
    classes_path = os.path.join(base_path, 'elliptic_txs_classes.csv')
    edgelist_path = os.path.join(base_path, 'elliptic_txs_edgelist.csv')
    features_path = os.path.join(base_path, 'elliptic_txs_features.csv')

    # Verify if files exist
    if not (os.path.exists(classes_path) and os.path.exists(edgelist_path) and os.path.exists(features_path)):
        raise FileNotFoundError("One or more required files are not found in the specified paths.")

    # Load CSV files
    try:
        classes_df = pd.read_csv(classes_path)
        edgelist_df = pd.read_csv(edgelist_path)
        features_df = pd.read_csv(features_path, header=None)
    except Exception as e:
        raise ValueError(f"Error loading CSV files: {e}")

    # Check and display null values in classes
    print("Checking null values in classes...")
    if classes_df.isnull().values.any():
        print(f"Null values found in classes. Number of null values: {classes_df.isnull().sum()}")
        # Show rows with null values
        print(classes_df[classes_df.isnull().any(axis=1)])

    # Remove transactions with 'unknown' class
    classes_df = classes_df[classes_df['class'] != 'unknown']
    print(f"After removing 'unknown' values, {len(classes_df)} transactions remain.")

    # Replace classes '2' with 1 and '0' with 0
    classes_df['class'] = classes_df['class'].map({'2': 1, '0': 0})

    # Debug: Verify edgelist_df columns
    print(f"Columns of edgelist_df: {edgelist_df.columns}")
    print(f"First rows of edgelist_df:\n{edgelist_df.head()}")

    # Check if columns txId1 and txId2 are present
    if 'txId1' not in edgelist_df.columns or 'txId2' not in edgelist_df.columns:
        raise KeyError("'txId1' or 'txId2' not found in edgelist_df columns.")

    # Check if nodes in the graph have an assigned class
    missing_nodes = list(set(edgelist_df['txId1'].unique()).union(set(edgelist_df['txId2'].unique())) - set(classes_df['txId'].values))
    print(f"Found {len(missing_nodes)} nodes in the graph without an assigned class.")

    # Assign default class (e.g., '0') to nodes without a class
    classes_df = classes_df.set_index('txId')
    classes_df = classes_df.reindex(edgelist_df['txId1'].unique().tolist() + edgelist_df['txId2'].unique().tolist(), fill_value=0).reset_index()

    # Check how many rows remain after removing transactions with 'unknown'
    print(f"After cleaning, {len(classes_df)} transactions remain.")

    # Check null values in the classes DataFrame after reindexing
    print(f"After assigning default values, {classes_df.isnull().sum().sum()} null values remain in classes.")

    # Remove rows with null values in the 'class' column
    classes_df.dropna(subset=['class'], inplace=True)

    # Check if there are remaining rows after removing nulls
    if classes_df.empty:
        raise ValueError("The classes DataFrame is empty after removing null rows.")

    # Normalize features
    features_array = features_df.values
    scaler = StandardScaler()
    features_normalized = scaler.fit_transform(features_array)

    # Create graph
    G = nx.Graph()
    edges = [(row['txId1'], row['txId2']) for _, row in edgelist_df.iterrows()]
    G.add_edges_from(edges)

    # Verify that the graph has nodes and edges
    if G.number_of_nodes() == 0:
        raise ValueError("The graph has no nodes.")
    if G.number_of_edges() == 0:
        raise ValueError("The graph has no edges.")

    # Assign normalized features to nodes
    node_list = list(G.nodes())
    for idx, node in enumerate(node_list):
        G.nodes[node]['feature'] = features_normalized[idx]

    # Assign labels to nodes
    labels_dict = {row['txId']: row['class'] for _, row in classes_df.iterrows()}
    for node in G.nodes():
        G.nodes[node]['label'] = labels_dict.get(node, 0)  # Use 0 as default class if no label exists

    # Check if processed graph already exists
    if os.path.exists('data/graph.pkl'):
        print("Graph already exists, loading...")
    else:
        # Save the processed graph
        os.makedirs('data', exist_ok=True)  # Create 'data' directory if it doesn't exist
        with open('data/graph.pkl', 'wb') as f:
            pickle.dump(G, f)
        print(f"Graph created with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges.")
