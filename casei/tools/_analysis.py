"""
Analysis utilities for edge confidence and differential interactions.
"""

import torch
import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix
from anndata import AnnData
from typing import List, Tuple, Dict, Optional

__all__ = ["contrast_gene_interactions", "store_edge_confidence_matrix"]


def contrast_gene_interactions(
    model,
    adata_filtered: AnnData,
    label_encoder,
    cond1: str,
    cond2: str,
    top_k: int = 20,
    min_abs_weight: float = 0.0
) -> Tuple[pd.DataFrame, pd.DataFrame, np.ndarray]:
    """
    Compute differential gene-gene interactions (W W^T)_cond1 - (W W^T)_cond2.
    """
    gene_names = adata_filtered.var_names.tolist()
    cond1_idx = label_encoder.transform([cond1])[0]
    cond2_idx = label_encoder.transform([cond2])[0]

    with torch.no_grad():
        W_WT = model.get_active_weights().cpu().numpy()

    W1, W2 = W_WT[cond1_idx], W_WT[cond2_idx]
    delta = W1 - W2

    i_indices, j_indices = np.triu_indices(len(gene_names), k=1)
    results = []
    for i, j in zip(i_indices, j_indices):
        delta_w = delta[i, j]
        if abs(delta_w) >= min_abs_weight:
            results.append({
                'gene_i': gene_names[i], 'gene_j': gene_names[j],
                'gene_i_idx': i, 'gene_j_idx': j,
                'delta_weight': delta_w,
                'weight_in_cond1': W1[i, j], 'weight_in_cond2': W2[i, j]
            })

    results_df = pd.DataFrame(results)
    if len(results_df) == 0: return pd.DataFrame(), pd.DataFrame(), delta

    results_df = results_df.sort_values('delta_weight', key=abs, ascending=False)
    pos_df = results_df[results_df['delta_weight'] > 0].head(top_k)
    neg_df = results_df[results_df['delta_weight'] < 0].head(top_k)

    return pos_df, neg_df, delta


def store_edge_confidence_matrix(
    adata: AnnData,
    test_results: List[Dict],
    uns_key: str = "edge_conf_matrix_sparse",
    keep_fraction: float = 0.1,
    obs_key: str = "n_high_conf_edges"
) -> AnnData:
    """
    Store top X% of edges by confidence as a sparse matrix in adata.uns.
    """
    edge_df = pd.DataFrame(test_results)
    obs_index_map = {str(idx): i for i, idx in enumerate(adata.obs.index)}
    rows, cols, data = [], [], []

    for cond, group in edge_df.groupby("true_label"):
        group_sorted = group.sort_values("confidence", ascending=False)
        n_keep = max(1, int(len(group_sorted) * keep_fraction))
        top_group = group_sorted.iloc[:n_keep]

        for _, row in top_group.iterrows():
            u, v = str(row["u"]), str(row["v"])
            if u in obs_index_map and v in obs_index_map:
                i, j = obs_index_map[u], obs_index_map[v]
                rows.extend([i, j]); cols.extend([j, i]); data.extend([row["confidence"], row["confidence"]])

    edge_matrix_sparse = coo_matrix((data, (rows, cols)), shape=(adata.n_obs, adata.n_obs)).tocsr()
    adata.uns[uns_key] = edge_matrix_sparse
    adata.obs[obs_key] = np.array(edge_matrix_sparse.sum(axis=1)).flatten()

    return adata