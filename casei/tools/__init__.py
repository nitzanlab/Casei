"""
Tools for Preprocessing, Training, and Analysis
=================================================

Utility functions for data preparation, model training, edge prediction,
and differential interaction analysis.
"""

from ._preprocessing import preprocess_adata, set_random_seed
from ._training import train_edge_predictor
from ._analysis import contrast_gene_interactions, store_edge_confidence_matrix

__all__ = [
    "preprocess_adata",
    "set_random_seed",
    "train_edge_predictor",
    "contrast_gene_interactions",
    "store_edge_confidence_matrix",
]
