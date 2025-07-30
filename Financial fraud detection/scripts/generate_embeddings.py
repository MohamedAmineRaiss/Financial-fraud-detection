import os
import pickle
import torch
import numpy as np
from node2vec import Node2Vec

def generate_embeddings():
    # Check if the graph file exists
    if not os.path.exists('data/graph.pkl'):
        raise FileNotFoundError("Graph file 'graph.pkl' not found. Make sure the data is processed.")

    # Load the graph from the file
    with open('data/graph.pkl', 'rb') as f:
        G = pickle.load(f)

    # Check if embeddings already exist
    if os.path.exists('data/embeddings.pt'):
        print("Embeddings have already been generated, loading from 'embeddings.pt'.")
        return torch.load('data/embeddings.pt')

    # Configure Node2Vec
    node2vec = Node2Vec(
        G,
        dimensions=128,       # Number of embedding dimensions
        walk_length=15,       # Length of random walks
        num_walks=20,         # Number of random walks per node
        p=1,                  # Return control parameter
        q=2,                  # Symmetry control parameter
        workers=10            # Number of threads to parallelize training
    )

    # Train Node2Vec
    print("Training the Node2Vec model...")
    model_node2vec = node2vec.fit()

    # Convert embeddings to tensor
    embeddings_array = np.array([model_node2vec.wv[str(node)] for node in G.nodes()])
    embeddings = torch.tensor(embeddings_array, dtype=torch.float)

    # Save generated embeddings
    os.makedirs('data', exist_ok=True)  # Ensure the 'data' directory exists
    torch.save(embeddings, 'data/embeddings.pt')

    print(f"Embeddings generated for {embeddings.shape[0]} nodes with {embeddings.shape[1]} dimensions.")
    return embeddings

# Run the function if this script is executed directly
if __name__ == "__main__":
    generate_embeddings()
