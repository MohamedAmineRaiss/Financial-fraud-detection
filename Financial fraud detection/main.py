import os
import torch
from scripts.data_preparation import prepare_data
from scripts.generate_embeddings import generate_embeddings
from scripts.data_balancing import balance_data
from scripts.hyperparameter_search import hyperparameter_search, train_model_with_params  # Import from the correct file
from scripts.visualize_embeddings import visualize_embeddings
from scripts.evaluate_model import evaluate_model

def main():
    print("Preparing data...")
    prepare_data()  # Calls the function to prepare the data

    print("Generating embeddings...")
    generate_embeddings()  # Calls the function to generate embeddings

    print("Balancing and preparing data...")
    data = balance_data()  # Calls the function to balance the data and return the `Data` object

    # Additional verification before continuing
    if data is None or not hasattr(data, 'x') or not hasattr(data, 'edge_index') or not hasattr(data, 'y'):
        raise ValueError("The processed data is invalid or incomplete.")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Step 1: Perform hyperparameter search
    print("Starting hyperparameter search...")
    param_grid = {
        'learning_rate': [0.001, 0.005, 0.01],
        'hidden_dim': [64, 128, 256],
        'dropout_rate': [0.2, 0.4, 0.6]
    }
    best_metrics = hyperparameter_search(data, param_grid, device)
    best_params = best_metrics['params']
    print(f"Optimal hyperparameters found: {best_params}")

    # Step 2: Retrain the model with the best hyperparameters
    print("Retraining the model with the best hyperparameters...")
    model = train_model_with_params(data, **best_params, device=device)

    # Step 3: Evaluate the model
    print("Evaluating the model...")
    evaluate_model(data, model)  # Evaluates the trained model with the data

    # Step 4: Visualize the embeddings
    print("Generating T-SNE visualization...")
    visualize_embeddings(data)

    print("Process complete. Results stored in the 'results' folder.")

if __name__ == "__main__":
    main()
