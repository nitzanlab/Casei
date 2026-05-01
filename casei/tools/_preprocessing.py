"""
Preprocessing utilities for spatial transcriptomics data.
"""

import random
import os
import numpy as np
import torch
import scanpy as sc
from scipy.sparse import issparse
from sklearn.neighbors import kneighbors_graph
from sklearn.preprocessing import LabelEncoder
from torch_geometric.data import Data
from anndata import AnnData
from typing import List, Tuple

__all__ = ["preprocess_adata", "set_random_seed"]


def set_random_seed(seed: int = 42) -> None:
    """
    Set random seed for reproducibility across all libraries.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Make PyTorch deterministic (slower but reproducible)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    os.environ["PYTHONHASHSEED"] = str(seed)


def preprocess_adata(
    adata: AnnData,
    n_neighbors: int = 5,
    use_highly_variable: bool = True,
    n_top_genes: int = 2000,
    sample_id_key: str = 'sample_id',
    condition_key: str = 'condition',
    spatial_key: str = 'spatial'
) -> Tuple[List[Data], LabelEncoder, AnnData]:
    """
    Preprocess AnnData object for edge prediction training.

    Creates graph representations where samples become graphs with cells
    as nodes and spatial neighbors as edges.
    """
    # Normalize and log-transform if not already done
    if 'log1p' not in adata.uns:
        sc.pp.normalize_total(adata, target_sum=1e4)
        sc.pp.log1p(adata)

    # Select highly variable genes
    if use_highly_variable:
        if 'highly_variable' not in adata.var.columns:
            sc.pp.highly_variable_genes(adata, n_top_genes=n_top_genes)
        adata_filtered = adata[:, adata.var['highly_variable']].copy()
    else:
        adata_filtered = adata.copy()

    # Encode condition labels
    adata_filtered.obs.index = adata_filtered.obs.index.astype(str)
    le = LabelEncoder()
    adata_filtered.obs['condition_encoded'] = le.fit_transform(
        adata_filtered.obs[condition_key]
    )

    # Create graph data for each sample
    data_list = []
    for sample_id in adata_filtered.obs[sample_id_key].unique():
        sample_adata = adata_filtered[
            adata_filtered.obs[sample_id_key] == sample_id
        ].copy()
        original_indices = sample_adata.obs.index.astype(str).to_numpy()

        if issparse(sample_adata.X):
            x = torch.tensor(sample_adata.X.toarray(), dtype=torch.float)
        else:
            x = torch.tensor(sample_adata.X, dtype=torch.float)

        # Build spatial graph
        spatial = sample_adata.obsm[spatial_key]
        adj = kneighbors_graph(
            spatial, n_neighbors=n_neighbors, mode='connectivity', include_self=False
        )
        edge_index = torch.tensor(np.array(adj.nonzero()), dtype=torch.long)

        condition_label = sample_adata.obs['condition_encoded'].iloc[0]
        edge_labels = torch.tensor(
            [condition_label] * edge_index.shape[1], dtype=torch.long
        )

        data = Data(
            x=x, edge_index=edge_index, y=edge_labels,
            sample_id=sample_id, original_indices=original_indices
        )
        data_list.append(data)

    return data_list, le, adata_filtered