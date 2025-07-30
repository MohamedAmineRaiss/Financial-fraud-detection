import os
import torch
import itertools
from scripts.data_balancing import balance_data
from torch_geometric.nn import SAGEConv
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, average_precision_score


# GraphSAGE model with adjustable hyperparameters
class GraphSAGE(torch.nn.Module):
    def __init__(self, in_channels, hidden_dim, out_channels, dropout_rate):
        super(GraphSAGE, self).__init__()
        self.conv1 = SAGEConv(in_channels, hidden_dim)
        self.conv2 = SAGEConv(hidden_dim, hidden_dim // 2)
        self.conv3 = SAGEConv(hidden_dim // 2, out_channels)
        self.dropout = torch.nn.Dropout(dropout_rate)

    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        x = self.dropout(x)
        x = F.relu(self.conv2(x, edge_index))
        x = self.dropout(x)
        x = self.conv3(x, edge_index)
        return x


def train_model_with_params(data, learning_rate, hidden_dim, dropout_rate, device):
    """
    Trains the GraphSAGE model with the specified hyperparameters.
    """
    model = GraphSAGE(data.num_features, hidden_dim, 2, dropout_rate).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    loss_fn = torch.nn.CrossEntropyLoss()

    best_val_loss = float('inf')
    patience, patience_counter = 10, 0
    best_model_state = None

    for epoch in range(1, 101):  # Adjust the number of epochs if necessary
        model.train()
        optimizer.zero_grad()
        out = model(data.x.to(device), data.edge_index.to(device))
        loss = loss_fn(out[data.train_mask], data.y[data.train_mask])
        loss.backward()
        optimizer.step()

        # Evaluate loss on the validation set
        model.eval()
        with torch.no_grad():
            val_out = model(data.x.to(device), data.edge_index.to(device))
            val_loss = loss_fn(val_out[data.val_mask], data.y[data.val_mask]).item()

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_model_state = model.state_dict()
        else:
            patience_counter += 1
            if patience_counter == patience:
                print(f"Early stopping at epoch {epoch}")
                break

    # Load the best model found during training
    if best_model_state:
        model.load_state_dict(best_model_state)

    return model


def hyperparameter_search(data, param_grid, device):
    """
    Performs hyperparameter search.
    """
    best_metrics = {'auc_roc': 0, 'params': None}
    results = []

    # Generate all hyperparameter combinations
    for params in itertools.product(*param_grid.values()):
        param_dict = dict(zip(param_grid.keys(), params))
        print(f"Testing hyperparameters: {param_dict}")

        # Train the model with current hyperparameters
        model = train_model_with_params(data, **param_dict, device=device)

        # Evaluate the model
        model.eval()
        with torch.no_grad():
            output = model(data.x.to(device), data.edge_index.to(device))
            auc_roc = roc_auc_score(data.y.cpu(), output.softmax(dim=1)[:, 1].cpu())
            auc_pr = average_precision_score(data.y.cpu(), output.softmax(dim=1)[:, 1].cpu())

        # Save results
        results.append({'params': param_dict, 'auc_roc': auc_roc, 'auc_pr': auc_pr})
        print(f"AUC-ROC: {auc_roc:.4f}, AUC-PR: {auc_pr:.4f}")

        # Update best hyperparameters
        if auc_roc > best_metrics['auc_roc']:
            best_metrics['auc_roc'] = auc_roc
            best_metrics['auc_pr'] = auc_pr
            best_metrics['params'] = param_dict

    # Sort results by descending AUC-ROC
    results = sorted(results, key=lambda x: x['auc_roc'], reverse=True)

    # Save results to a file
    output_file = "results/hyperparameter_results.txt"
    os.makedirs("results", exist_ok=True)
    with open(output_file, "w") as f:
        for result in results:
            f.write(f"Params: {result['params']}, AUC-ROC: {result['auc_roc']:.4f}, AUC-PR: {result['auc_pr']:.4f}\n")

    print(f"Best hyperparameters: {best_metrics['params']}")
    print(f"Results saved to '{output_file}'")

    return best_metrics


if __name__ == "__main__":
    # Get balanced data
    data = balance_data()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Define hyperparameter search space
    param_grid = {
        'learning_rate': [0.001, 0.005, 0.01],
        'hidden_dim': [64, 128, 256],
        'dropout_rate': [0.2, 0.4, 0.6]
    }

    # Run hyperparameter search
    best_metrics = hyperparameter_search(data, param_grid, device)

    print("Optimal hyperparameters found:")
    print(best_metrics)
