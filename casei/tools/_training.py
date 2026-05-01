"""
Training utilities for edge prediction models.
"""

import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from typing import List, Dict, Tuple, Optional
from torch_geometric.data import Data
from anndata import AnnData
from sklearn.preprocessing import LabelEncoder

# Relative import for your EdgePredictionMLP model
from ..models import EdgePredictionMLP

__all__ = ["train_edge_predictor"]


def train_edge_predictor(
    data_list: List[Data],
    adata_filtered: AnnData,
    label_encoder: LabelEncoder,
    in_channels: int,
    hidden_channels: int = 64,
    epochs: int = 50,
    lr: float = 1e-4,
    lambda_l1: float = 1e-5,
    lambda_l2: float = 1e-4,
    device: Optional[str] = None
) -> Tuple[EdgePredictionMLP, List[Dict]]:
    """
    Train edge prediction model with L1/L2 regularization.
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(device)

    model = EdgePredictionMLP(
        in_channels=in_channels,
        hidden_channels=hidden_channels,
        out_channels=len(label_encoder.classes_)
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = torch.nn.CrossEntropyLoss()
    probabilities = []

    # Training loop
    for epoch in tqdm(range(epochs), desc="Training"):
        model.train()
        total_class_loss = 0
        total_reg_loss = 0

        for data in data_list:
            data = data.to(device)
            optimizer.zero_grad()

            out = model(data.x, data.edge_index)
            class_loss = criterion(out, data.y)
            reg_loss = model.get_regularization_loss(
                lambda_l1=lambda_l1, lambda_l2=lambda_l2
            )

            loss = class_loss + reg_loss
            loss.backward()
            optimizer.step()

            total_class_loss += class_loss.item()
            total_reg_loss += reg_loss.item()

        if epoch % 10 == 0:
            tqdm.write(f"Epoch {epoch}: Class Loss={total_class_loss:.4f}, Reg Loss={total_reg_loss:.4f}")

        # Collect validation probabilities for confidence calculation
        model.eval()
        epoch_probs = []
        with torch.no_grad():
            for data in data_list:
                data = data.to(device)
                out = model(data.x, data.edge_index)
                probs = F.softmax(out, dim=1)
                true_class_probs = probs[range(probs.shape[0]), data.y]
                epoch_probs.append(true_class_probs.cpu().numpy())
        probabilities.append(epoch_probs)

    # Collect test results using averaged confidence
    test_results = []
    model.eval()
    with torch.no_grad():
        for data_idx, data in enumerate(data_list):
            data = data.to(device)
            original_indices = data.original_indices
            edge_index = data.edge_index.cpu().numpy()
            out = model(data.x, data.edge_index)
            pred = out.argmax(dim=1).cpu().numpy()
            true_y = data.y.cpu().numpy()

            # Average confidence over all epochs
            epoch_probs_arr = np.array([probabilities[e][data_idx] for e in range(epochs)])
            edge_confidence = np.mean(epoch_probs_arr, axis=0)

            for i, (u, v) in enumerate(edge_index.T):
                test_results.append({
                    'sample_id': data.sample_id,
                    'u': original_indices[u],
                    'v': original_indices[v],
                    'confidence': edge_confidence[i],
                    'prediction': label_encoder.inverse_transform([pred[i]])[0],
                    'true_label': label_encoder.inverse_transform([true_y[i]])[0]
                })

    return model, test_results