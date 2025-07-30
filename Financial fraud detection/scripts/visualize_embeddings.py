import os
import torch
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from scripts.data_balancing import balance_data
import seaborn as sns
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D


# Create the results folder if it doesn't exist
os.makedirs("results", exist_ok=True)

def visualize_embeddings(data):
    """
    Visualizes embeddings using multiple techniques.
    
    Args:
        data: PyTorch Geometric `Data` object containing embeddings and labels.
    """
    print("Generating multiple embedding visualizations...")
    
    # Verify that the data has embeddings and labels
    if not hasattr(data, 'x') or not hasattr(data, 'y'):
        raise ValueError("The `data` object does not contain the required attributes (`x` and `y`).")
    
    embeddings = data.x.cpu().detach().numpy()  # Convert embeddings to numpy
    labels = data.y.cpu().numpy()  # Labels
    
    # Normalize embeddings before applying dimensionality reduction
    scaler = StandardScaler()
    normalized_embeddings = scaler.fit_transform(embeddings)
    
    # 1. T-SNE 2D Visualization
    print("Generating T-SNE 2D visualization...")
    tsne = TSNE(n_components=2, random_state=42, perplexity=30, learning_rate=200)
    reduced_embeddings_tsne = tsne.fit_transform(normalized_embeddings)
    
    plt.figure(figsize=(12, 8))
    scatter = plt.scatter(
        reduced_embeddings_tsne[:, 0],
        reduced_embeddings_tsne[:, 1],
        c=labels,
        cmap="coolwarm",
        alpha=0.6,
        edgecolors="k",
        s=15
    )
    plt.colorbar(scatter, label="Classes (0 = Legit, 1 = Fraud)")
    plt.title("T-SNE 2D Visualization of Embeddings")
    plt.xlabel("T-SNE Component 1")
    plt.ylabel("T-SNE Component 2")
    
    # Save visualization in the `results` folder
    output_path = "results/tsne_2d_visualization.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"T-SNE 2D visualization saved to '{output_path}'.")
    
    # 2. T-SNE 3D Visualization
    print("Generating T-SNE 3D visualization...")
    tsne_3d = TSNE(n_components=3, random_state=42, perplexity=30, learning_rate=200)
    reduced_embeddings_tsne_3d = tsne_3d.fit_transform(normalized_embeddings)
    
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    scatter = ax.scatter(
        reduced_embeddings_tsne_3d[:, 0],
        reduced_embeddings_tsne_3d[:, 1],
        reduced_embeddings_tsne_3d[:, 2],
        c=labels,
        cmap="coolwarm",
        alpha=0.6,
        edgecolors="k",
        s=15
    )
    
    plt.colorbar(scatter, label="Classes (0 = Legit, 1 = Fraud)")
    ax.set_title("T-SNE 3D Visualization of Embeddings")
    ax.set_xlabel("T-SNE Component 1")
    ax.set_ylabel("T-SNE Component 2")
    ax.set_zlabel("T-SNE Component 3")
    
    # Save 3D visualization
    output_path_3d = "results/tsne_3d_visualization.png"
    plt.savefig(output_path_3d, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"T-SNE 3D visualization saved to '{output_path_3d}'.")
    
    # 3. PCA Visualization
    print("Generating PCA visualization...")
    pca = PCA(n_components=2)
    reduced_embeddings_pca = pca.fit_transform(normalized_embeddings)
    
    plt.figure(figsize=(12, 8))
    scatter = plt.scatter(
        reduced_embeddings_pca[:, 0],
        reduced_embeddings_pca[:, 1],
        c=labels,
        cmap="coolwarm",
        alpha=0.6,
        edgecolors="k",
        s=15
    )
    plt.colorbar(scatter, label="Classes (0 = Legit, 1 = Fraud)")
    plt.title("PCA Visualization of Embeddings")
    plt.xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)")
    plt.ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.2%} variance)")
    
    # Save PCA visualization
    output_path_pca = "results/pca_visualization.png"
    plt.savefig(output_path_pca, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"PCA visualization saved to '{output_path_pca}'.")
    
  
    
    # 5. Class-specific T-SNE Visualization
    print("Generating class-specific T-SNE visualization...")
    plt.figure(figsize=(16, 8))
    
    # Legitimate transactions
    plt.subplot(1, 2, 1)
    legit_indices = np.where(labels == 0)[0]
    plt.scatter(
        reduced_embeddings_tsne[legit_indices, 0],
        reduced_embeddings_tsne[legit_indices, 1],
        c='blue',
        alpha=0.6,
        edgecolors="k",
        s=15,
        label="Legitimate"
    )
    plt.title("T-SNE: Legitimate Transactions")
    plt.xlabel("T-SNE Component 1")
    plt.ylabel("T-SNE Component 2")
    plt.legend()
    
    # Fraudulent transactions
    plt.subplot(1, 2, 2)
    fraud_indices = np.where(labels == 1)[0]
    plt.scatter(
        reduced_embeddings_tsne[fraud_indices, 0],
        reduced_embeddings_tsne[fraud_indices, 1],
        c='red',
        alpha=0.6,
        edgecolors="k",
        s=15,
        label="Fraud"
    )
    plt.title("T-SNE: Fraudulent Transactions")
    plt.xlabel("T-SNE Component 1")
    plt.ylabel("T-SNE Component 2")
    plt.legend()
    
    plt.tight_layout()
    
    # Save class-specific visualization
    output_path_class = "results/tsne_class_specific.png"
    plt.savefig(output_path_class, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Class-specific T-SNE visualization saved to '{output_path_class}'.")
    
    # 6. Density Plot of Embeddings
    print("Generating density plot of embeddings...")
    plt.figure(figsize=(16, 8))
    
    # Legitimate transactions
    plt.subplot(1, 2, 1)
    sns.kdeplot(
        x=reduced_embeddings_tsne[legit_indices, 0],
        y=reduced_embeddings_tsne[legit_indices, 1],
        cmap="Blues",
        fill=True,
        thresh=0.05
    )
    plt.title("Density: Legitimate Transactions")
    plt.xlabel("T-SNE Component 1")
    plt.ylabel("T-SNE Component 2")
    
    # Fraudulent transactions
    plt.subplot(1, 2, 2)
    sns.kdeplot(
        x=reduced_embeddings_tsne[fraud_indices, 0],
        y=reduced_embeddings_tsne[fraud_indices, 1],
        cmap="Reds",
        fill=True,
        thresh=0.05
    )
    plt.title("Density: Fraudulent Transactions")
    plt.xlabel("T-SNE Component 1")
    plt.ylabel("T-SNE Component 2")
    
    plt.tight_layout()
    
    # Save density plot
    output_path_density = "results/tsne_density_plot.png"
    plt.savefig(output_path_density, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Density plot saved to '{output_path_density}'.")
    
    print("All embedding visualizations completed.")

if __name__ == "__main__":
    # Get balanced data
    data = balance_data()
    
    # Call visualization function
    visualize_embeddings(data)
