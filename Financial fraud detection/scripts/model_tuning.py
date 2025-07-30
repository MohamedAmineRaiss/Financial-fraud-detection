import os
import torch
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_curve, auc, roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns
from torch_geometric.nn import SAGEConv
from scripts.train_model import train_model
from scripts.evaluate_model import evaluate_model

# Create the results folder if it doesn't exist
os.makedirs("results", exist_ok=True)

def evaluate_with_metrics(model, data, device):
    # Set the model to evaluation mode
    model.eval()
    
    # Make predictions: pass the complete data object
    output = model(data.to(device))  # Send the entire 'data' object to the model
    
    # Get predictions
    pred = output.argmax(dim=1)

    # Calculate metrics
    print("Evaluating the model...")

    # Get true labels
    true_labels = data.y.to(device)

    # Classification report (Precision, Recall, F1, etc.)
    print("Classification report:")
    print(classification_report(true_labels.cpu(), pred.cpu(), target_names=["Legit", "Fraud"]))

    # Confusion matrix
    print("Confusion Matrix:")
    cm = confusion_matrix(true_labels.cpu(), pred.cpu())
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=["Legit", "Fraud"], yticklabels=["Legit", "Fraud"])
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.savefig("results/confusion_matrix.png")  # Save the plot
    plt.close()  # Close the plot to free memory

    # Precision-Recall curve
    precision, recall, _ = precision_recall_curve(true_labels.cpu(), output[:, 1].cpu().detach().numpy())
    pr_auc = auc(recall, precision)
    print(f"AUC-PR: {pr_auc:.4f}")
    plt.figure()
    plt.plot(recall, precision, label=f'AUC-PR = {pr_auc:.4f}')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend()
    plt.savefig("results/precision_recall_curve.png")  # Save the plot
    plt.close()

    # AUC-ROC
    auc_roc = roc_auc_score(true_labels.cpu(), output[:, 1].cpu().detach().numpy())
    print(f"AUC-ROC: {auc_roc:.4f}")

    # ROC curve
    fpr, tpr, _ = precision_recall_curve(true_labels.cpu(), output[:, 1].cpu().detach().numpy())
    plt.figure()
    plt.plot(fpr, tpr, label=f'AUC-ROC = {auc_roc:.4f}')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend()
    plt.savefig("results/roc_curve.png")  # Save the plot
    plt.close()

    return auc_roc, pr_auc

def tune_model(data):
    # Check device (GPU if available)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Train the model with balanced data
    model = train_model(data)  # This method should train and return the trained model
    
    # Evaluate the model and get metrics
    auc_roc, pr_auc = evaluate_with_metrics(model, data, device)

    return model, auc_roc, pr_auc

if __name__ == "__main__":
    # Run model tuning and evaluation
    from scripts.data_balancing import balance_data
    data = balance_data()  # Get balanced data
    
    # Tune and evaluate the model
    model, auc_roc, pr_auc = tune_model(data)
    
    # Print metrics
    print(f"AUC-ROC: {auc_roc:.4f}")
    print(f"AUC-PR: {pr_auc:.4f}")
