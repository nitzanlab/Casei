"""
Casei: Spatial Transcriptomics Edge Prediction and Analysis
=============================================================

A Python package for analyzing spatial gene expression data using graph neural networks
for edge prediction and differential interaction analysis.

Modules:
--------
- models: Neural network models for edge prediction
- plotting: Visualization functions for spatial data and interactions
- tools: Utility functions for preprocessing, analysis, and graph operations
"""

from ._version import __version__, __version_info__

__author__ = "Casei Development Team"

# Import main modules
from . import models
from . import plotting as pl
from . import tools as tl

# Import key classes and functions for convenience
from .models import EdgePredictionMLP
from .tools import (
    preprocess_adata,
    set_random_seed,
    train_edge_predictor,
    contrast_gene_interactions,
    store_edge_confidence_matrix,
)
from .plotting import (
    plot_differential_heatmap,
    plot_gene_network,
    plot_edge_comparison,
    plot_edges_per_sample,
)

__all__ = [
    "models",
    "pl",
    "tl",
    "EdgePredictionMLP",
    "preprocess_adata",
    "set_random_seed",
    "train_edge_predictor",
    "contrast_gene_interactions",
    "store_edge_confidence_matrix",
    "plot_differential_heatmap",
    "plot_gene_network",
    "plot_edge_comparison",
    "plot_edges_per_sample",
]
