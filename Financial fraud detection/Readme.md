# **Network Analysis and Fraud Detection Using Graph Neural Networks**

This project leverages **Graph Neural Networks (GNNs)** for detecting fraudulent transactions in financial transaction networks. It includes a modular implementation of models like **Graph Attention Networks (GAT)**, **Graph Autoencoders (GAE)**, and **GraphSAGE**, along with preprocessing, embedding generation, and model evaluation.

---

## **Table of Contents**
1. [Features](#features)
2. [Installation](#installation)
3. [Usage](#usage)
4. [Project Structure](#project-structure)
5. [Results](#results)
6. [References](#references)

---

## **Features**

- **Preprocessing**: Transforms financial transaction data into graph-structured data.
- **Embedding Generation**: Supports GAT, GAE, and GraphSAGE-based embeddings.
- **Visualization**: T-SNE-based embedding visualization.
- **Evaluation**: Performance metrics like AUC-ROC, AUC-PR, and F1-Score.
- **Hyperparameter Tuning**: Grid search for optimal parameters.

---

## **Installation**

### **Prerequisites**
- Python 3.8+
- PyTorch (with CUDA support, if available)
- PyTorch Geometric

### **Setup**

1. Clone the repository:
   ```bash
   git clone https://github.com/ushio2580/gnn-fraud-detection.git
   cd gnn-fraud-detection
   ```

2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Set up the dataset:
   - Download the **Elliptic Bitcoin Dataset** from [Elliptic Co](https://www.elliptic.co/).
   - Extract the dataset to `data/elliptic_bitcoin_dataset`.

4. Prepare the data:
   ```bash
   python3 scripts/data_preparation.py
   ```

---

## **Usage**

### **Step 1: Generate Embeddings**
Generate initial embeddings using Node2Vec:
```bash
python3 scripts/generate_embeddings.py
```

### **Step 2: Train and Evaluate Models**
1. Perform data balancing and prepare for training:
   ```bash
   python3 scripts/data_balancing.py
   ```

2. Train models with hyperparameter tuning:
   ```bash
   python3 scripts/hyperparameter_search.py
   ```

3. Evaluate the best-performing model:
   ```bash
   python3 scripts/evaluate_model.py
   ```

### **Step 3: Visualize Embeddings**
Visualize embeddings using T-SNE:
```bash
python3 scripts/visualize_embeddings.py
```

### **Run Full Pipeline**
To run the entire pipeline end-to-end:
```bash
python3 main.py
```

---

## **Project Structure**

```plaintext
gnn-fraud-detection/
│
├── data/                          # Dataset and processed data
├── results/                       # Generated results (e.g., metrics, visualizations)
├── models/                        # Saved models
├── scripts/                       # Scripts for individual tasks
│   ├── data_preparation.py        # Preprocessing and graph construction
│   ├── generate_embeddings.py     # Node2Vec-based embedding generation
│   ├── data_balancing.py          # Balancing classes with SMOTE
│   ├── train_model.py             # Training GNN models
│   ├── hyperparameter_search.py   # Hyperparameter tuning
│   ├── evaluate_model.py          # Model evaluation
│   ├── visualize_embeddings.py    # T-SNE visualization
│   └── main.py                    # Full pipeline execution
├── requirements.txt               # List of Python dependencies
└── README.md                      # Project documentation
```

---

## **Results**

### **Best Model Configuration**
| **Learning Rate** | **Hidden Dimensions** | **Dropout** | **AUC-ROC** | **AUC-PR** | **F1 Score** |
|-------------------|-----------------------|-------------|-------------|------------|--------------|
| 0.01              | 256                   | 0.4         | **0.8190**  | **0.8135** | **0.7991**   |

### **Visualization**
The T-SNE visualization below shows the separation between legitimate and fraudulent transactions:

![T-SNE Visualization](results/tsne_visualization.png)

### **Model Performance Comparison**
| **Model**   | **AUC-ROC** | **AUC-PR** |
|-------------|-------------|------------|
| GAT         | 0.8156      | 0.7943     |
| GAE         | 0.8078      | 0.7808     |
| GraphSAGE   | **0.8190**  | **0.8135** |

---

## **References**

1. Kipf, T. N., & Welling, M. (2016). Semi-Supervised Classification with Graph Convolutional Networks. *arXiv preprint arXiv:1609.02907.*
2. Velickovic, P., et al. (2017). Graph Attention Networks. *arXiv preprint arXiv:1710.10903.*
3. Chawla, N. V., et al. (2002). SMOTE: Synthetic Minority Over-sampling Technique. *Journal of Artificial Intelligence Research, 16, 321–357.*
4. Maaten, L., & Hinton, G. (2008). Visualizing Data using t-SNE. *Journal of Machine Learning Research, 9(Nov), 2579–2605.*

---

## **License**
This project is licensed under the MIT License.
