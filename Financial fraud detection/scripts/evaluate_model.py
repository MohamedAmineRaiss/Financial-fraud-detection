import torch
from torch_geometric.nn import SAGEConv
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, f1_score, confusion_matrix, precision_recall_curve, roc_curve, auc, precision_score, recall_score
import matplotlib.pyplot as plt
import seaborn as sns
import os
import numpy as np
import pandas as pd

# Create folder to save results
os.makedirs('results', exist_ok=True)

# GraphSAGE model (identical to the training one)
class GraphSAGE(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(GraphSAGE, self).__init__()
        self.conv1 = SAGEConv(in_channels, 256)
        self.conv2 = SAGEConv(256, 128)
        self.conv3 = SAGEConv(128, 64)
        self.conv4 = SAGEConv(64, out_channels)
        self.dropout = torch.nn.Dropout(0.4)

    # Changed to explicitly accept x and edge_index
    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        x = self.dropout(x)
        x = F.relu(self.conv2(x, edge_index))
        x = self.dropout(x)
        x = F.relu(self.conv3(x, edge_index))
        x = self.dropout(x)
        x = self.conv4(x, edge_index)
        return x

def evaluate_model(data, model=None):
    try:
        # Load model if not passed as argument
        if model is None:
            model = GraphSAGE(data.x.size(1), 2).to(data.x.device)
            model.load_state_dict(torch.load('models/graphsage_best_model.pth'))
            print("Model loaded successfully.")

        model.eval()

        # Make predictions (changed to pass x and edge_index explicitly)
        with torch.no_grad():
            out = model(data.x, data.edge_index)  # Changed to pass attributes explicitly
            prob = F.softmax(out, dim=1)
            pred = torch.argmax(prob, dim=1)

        # Convert to numpy for metrics calculation
        y_true = data.y.cpu().numpy()
        y_pred = pred.cpu().numpy()
        y_prob = prob[:, 1].cpu().numpy()  # Probability of fraud class

        # Ensure there are at least two classes in the predictions
        unique_preds = torch.unique(pred)
        if len(unique_preds) == 1:
            print("Warning: Predictions contain only one class. Cannot calculate AUC-ROC.")
            auc_score = None
        else:
            # Calculate metrics if there are at least two classes
            auc_score = roc_auc_score(y_true, y_prob)
            print(f"AUC-ROC: {auc_score:.4f}")

        # Calculate additional metrics
        f1 = f1_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)
        
        print(f"F1-Score: {f1:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")

        # 1. Confusion Matrix
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.title("Confusion Matrix")
        plt.savefig('results/confusion_matrix.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. Normalized Confusion Matrix
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Blues')
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.title("Normalized Confusion Matrix")
        plt.savefig('results/confusion_matrix_normalized.png', dpi=300, bbox_inches='tight')
        plt.close()

        # 3. ROC Curve
        if auc_score is not None:
            fpr, tpr, _ = roc_curve(y_true, y_prob)
            plt.figure(figsize=(10, 8))
            plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {auc_score:.2f})')
            plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title('Receiver Operating Characteristic (ROC) Curve')
            plt.legend(loc="lower right")
            plt.savefig('results/roc_curve.png', dpi=300, bbox_inches='tight')
            plt.close()

        # 4. Precision-Recall Curve
        if len(np.unique(y_true)) > 1:
            precision_curve, recall_curve, _ = precision_recall_curve(y_true, y_prob)
            pr_auc = auc(recall_curve, precision_curve)
            
            plt.figure(figsize=(10, 8))
            plt.plot(recall_curve, precision_curve, color='blue', lw=2, 
                    label=f'PR curve (area = {pr_auc:.2f})')
            plt.xlabel('Recall')
            plt.ylabel('Precision')
            plt.title('Precision-Recall Curve')
            plt.legend(loc="lower left")
            plt.savefig('results/pr_curve.png', dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"PR-AUC: {pr_auc:.4f}")

        # 5. Class-specific metrics bar chart
        plt.figure(figsize=(12, 8))
        metrics_df = pd.DataFrame({
            'Precision': [precision],
            'Recall': [recall],
            'F1 Score': [f1]
        }, index=['Fraud Detection'])
        metrics_df.plot(kind='bar', figsize=(12, 8))
        plt.title('Performance Metrics')
        plt.ylabel('Score')
        plt.ylim([0, 1])
        plt.xticks(rotation=0)
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.savefig('results/performance_metrics.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 6. Error Analysis: Misclassification Distribution
        misclassified = y_pred != y_true
        misclassified_indices = np.where(misclassified)[0]
        
        if len(misclassified_indices) > 0:
            misclassified_true = y_true[misclassified_indices]
            misclassified_pred = y_pred[misclassified_indices]
            
            # Count false positives and false negatives
            fp = np.sum((misclassified_true == 0) & (misclassified_pred == 1))
            fn = np.sum((misclassified_true == 1) & (misclassified_pred == 0))
            
            plt.figure(figsize=(10, 8))
            plt.bar(['False Positives', 'False Negatives'], [fp, fn], color=['skyblue', 'salmon'])
            plt.title('Error Distribution')
            plt.ylabel('Count')
            plt.grid(axis='y', linestyle='--', alpha=0.7)
            plt.savefig('results/error_distribution.png', dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"False Positives: {fp}")
            print(f"False Negatives: {fn}")

        # Save metrics to text file
        with open('results/metrics.txt', 'w') as f:
            f.write(f"F1-Score: {f1:.4f}\n")
            f.write(f"Precision: {precision:.4f}\n")
            f.write(f"Recall: {recall:.4f}\n")
            if auc_score is not None:
                f.write(f"AUC-ROC: {auc_score:.4f}\n")
            if 'pr_auc' in locals():
                f.write(f"PR-AUC: {pr_auc:.4f}\n")
            f.write(f"False Positives: {fp if 'fp' in locals() else 0}\n")
            f.write(f"False Negatives: {fn if 'fn' in locals() else 0}\n")
        
        # Save metrics to JSON for easier parsing
        import json
        metrics_dict = {
            'f1_score': float(f1),
            'precision': float(precision),
            'recall': float(recall),
            'false_positives': int(fp) if 'fp' in locals() else 0,
            'false_negatives': int(fn) if 'fn' in locals() else 0
        }
        
        if auc_score is not None:
            metrics_dict['auc_roc'] = float(auc_score)
        
        if 'pr_auc' in locals():
            metrics_dict['pr_auc'] = float(pr_auc)
            
        with open('results/metrics.json', 'w') as f:
            json.dump(metrics_dict, f, indent=4)
            
        print("Evaluation complete. Results saved to 'results/' directory.")

    except Exception as e:
        print(f"Error in evaluation: {e}")
        import traceback
        traceback.print_exc()
