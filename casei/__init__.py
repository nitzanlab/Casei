"""
Casei: Cellular Interaction Spatial Evolution & Inference
==========================================================

A Python package for analyzing spatial gene expression data using graph neural networks
for edge prediction, differential interaction analysis, and program enrichment.

Modules:
--------
- tl (tools): Utility functions for preprocessing, analysis, and graph operations
- pl (plotting): Visualization functions for spatial data and interactions
- enr (enrichment): Eigendecomposition-based program and GO/pathway enrichment
- models: Neural network models for edge prediction
"""

from ._version import __version__, __version_info__

__author__ = "Casei Development Team"

# Import main modules with standard aliases
from . import models
from . import tools as tl
from . import plotting as pl
from . import enrichment as enr

# Import key classes and functions for convenience
from .models import EdgePredictionMLP

# Tools
from .tools import (
    preprocess_adata,
    set_random_seed,
    train_edge_predictor,
    contrast_gene_interactions,
    store_edge_confidence_matrix,
)

# Plotting
from .plotting import (
    plot_differential_heatmap,
    plot_gene_network,
    plot_edge_comparison,
    plot_edges_per_sample,
    plot_edges_on_umap,
    plot_edges_grid_umap,
    compare_conditions_edges_umap,
    compare_neighborhood_enrichment,
    analyze_interaction_drivers,
)

# Enrichment
from .enrichment import (
    run_enrichment_analysis,
    decompose_differential_matrix,
    contrast_gene_gene_interactions,
)

__all__ = [
    "models",
    "tl",
    "pl",
    "enr",
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
    "plot_edges_on_umap",
    "plot_edges_grid_umap",
    "compare_conditions_edges_umap",
    "compare_neighborhood_enrichment",
    "analyze_interaction_drivers",
    "run_enrichment_analysis",
    "decompose_differential_matrix",
    "contrast_gene_gene_interactions",
]