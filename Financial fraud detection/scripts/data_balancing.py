import os
import pickle
import numpy as np
import torch
from torch_geometric.data import Data
from imblearn.over_sampling import SMOTE

def balance_data():
    # Load the graph
    with open('data/graph.pkl', 'rb') as f:
        G = pickle.load(f)

    # Verify that the graph has nodes and labels
    if len(G.nodes()) == 0:
        raise ValueError("The graph is empty.")
    
    missing_labels = [node for node in G.nodes() if 'label' not in G.nodes[node]]
    if missing_labels:
        print(f"The following nodes have no label: {missing_labels}")

    # Verify the labels of nodes in the graph
    labels = [G.nodes[node].get('label', 0.0) for node in G.nodes()]
    print(f"Label distribution in the graph: {set(labels)}")

    # Remove nodes with null or unknown labels
    valid_nodes = [node for node in G.nodes() if G.nodes[node].get('label') not in [None, 'unknown']]
    G = G.subgraph(valid_nodes)
    print(f"After cleaning, {len(G.nodes())} nodes remain in the graph.")

    # Load embeddings
    embeddings = torch.load('data/embeddings.pt')

    # Verify that embeddings have the correct shape
    if embeddings.shape[0] != len(G.nodes()):
        raise ValueError(f"Embeddings have incorrect size: {embeddings.shape[0]} nodes, but the graph has {len(G.nodes())} nodes.")

    # Label assignment (adjusted according to observed distribution)
    labels_tensor = torch.tensor(
        [1 if G.nodes[node].get('label', 0.0) == 1.0 else 0 for node in G.nodes()],
        dtype=torch.long
    )

    # Verify that labels have the same length as embeddings
    if len(labels_tensor) != len(embeddings):
        raise ValueError(f"Labels do not match embeddings. They have {len(labels_tensor)} and {len(embeddings)} elements respectively.")

    # Print number of nodes in each class
    print(f"'fraud' labels: {labels_tensor.sum().item()}")
    print(f"'legit' labels: {(labels_tensor == 0).sum().item()}")

    # Class balancing using SMOTE
    fraud_indices = [i for i, label in enumerate(labels_tensor) if label == 1]
    legit_indices = [i for i, label in enumerate(labels_tensor) if label == 0]

    # Print number of nodes in each class before balancing
    print(f"Total 'fraud' class nodes: {len(fraud_indices)}")
    print(f"Total 'legit' class nodes: {len(legit_indices)}")

    if len(fraud_indices) == 0 or len(legit_indices) == 0:
        raise ValueError("Not enough nodes of one class for balancing.")

    # Apply SMOTE: we'll use only the embeddings (features) for creating synthetic examples
    smote = SMOTE(sampling_strategy='auto', random_state=42)
    embeddings_np = embeddings.cpu().numpy()  # Convert to numpy for use with SMOTE
    labels_np = labels_tensor.cpu().numpy()

    # Apply SMOTE on the features (embeddings) of the nodes
    embeddings_resampled, labels_resampled = smote.fit_resample(embeddings_np, labels_np)

    # Convert back to PyTorch tensors
    embeddings_resampled = torch.tensor(embeddings_resampled, dtype=torch.float)
    labels_resampled = torch.tensor(labels_resampled, dtype=torch.long)

    # Verify balancing after applying SMOTE
    print(f"Balance after SMOTE:")
    print(f"'fraud' labels: {labels_resampled.sum().item()}")
    print(f"'legit' labels: {(labels_resampled == 0).sum().item()}")

    # Create splits to ensure both classes are present in both sets
    fraud_indices_resampled = [i for i, label in enumerate(labels_resampled) if label == 1]
    legit_indices_resampled = [i for i, label in enumerate(labels_resampled) if label == 0]

    # Split the data to ensure both sets have both classes
    train_fraud_indices = fraud_indices_resampled[:int(0.8 * len(fraud_indices_resampled))]
    val_fraud_indices = fraud_indices_resampled[int(0.8 * len(fraud_indices_resampled)):]

    train_legit_indices = legit_indices_resampled[:int(0.8 * len(legit_indices_resampled))]
    val_legit_indices = legit_indices_resampled[int(0.8 * len(legit_indices_resampled)):]

    # Combine splits for training and validation
    train_indices = train_fraud_indices + train_legit_indices
    val_indices = val_fraud_indices + val_legit_indices

    # Create masks for PyTorch Geometric
    train_mask = torch.zeros(len(labels_resampled), dtype=torch.bool)
    val_mask = torch.zeros(len(labels_resampled), dtype=torch.bool)

    train_mask[train_indices] = True
    val_mask[val_indices] = True

    # Additional verification of class balance after balancing
    train_labels = labels_resampled[train_indices]
    val_labels = labels_resampled[val_indices]

    # Verify that both classes are present in training and validation data
    if len(train_labels.unique()) != 2:
        raise ValueError("The training set has only one class, cannot train a model with only one class.")
    
    if len(val_labels.unique()) != 2:
        raise ValueError("The validation set has only one class, cannot evaluate a model with only one class.")

    # Create edge_index for PyTorch Geometric
    node_mapping = {node: idx for idx, node in enumerate(G.nodes())}
    edge_index = torch.tensor(
        [[node_mapping[edge[0]], node_mapping[edge[1]]] for edge in G.edges()],
        dtype=torch.long
    ).t().contiguous()

    # Verify that edge_index has the correct shape
    if edge_index.shape[0] != 2:
        raise ValueError(f"edge_index has an incorrect shape: {edge_index.shape}")

    # Prepare data
    data = Data(
        x=embeddings_resampled,
        edge_index=edge_index,
        y=labels_resampled,
        train_mask=train_mask,
        val_mask=val_mask
    ).to('cuda' if torch.cuda.is_available() else 'cpu')

    # Verify that the prepared data has the correct shapes
    if not hasattr(data, 'x') or not hasattr(data, 'edge_index') or not hasattr(data, 'y'):
        raise ValueError("The 'Data' object has invalid or incomplete attributes.")

    # Save the processed data
    torch.save(data, 'data/processed_data.pt')
    print(f"Balanced and processed data saved to 'data/processed_data.pt'.")

    # Return the `data` object to be used in `main.py`
    return data

# Run the function if this script is executed directly
if __name__ == "__main__":
    balance_data()
