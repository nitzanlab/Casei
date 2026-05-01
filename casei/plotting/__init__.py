"""
Plotting Functions for Spatial Transcriptomics Analysis
========================================================

This module provides visualization functions for spatial data, differential
interactions, and network analysis.
"""

# Import core plotting implementations
from ._core import (
    plot_differential_heatmap,
    plot_gene_network,
    plot_edge_comparison,
    plot_edges_on_umap,
    plot_edges_grid_umap,
    compare_conditions_edges_umap,
    compare_neighborhood_enrichment,
)

# Import specialized functions from other modules
from .spatial_edges import plot_edges_per_sample
from .interaction_drivers import analyze_interaction_drivers

__all__ = [
    'plot_differential_heatmap',
    'plot_gene_network',
    'plot_edge_comparison',
    'plot_edges_on_umap',
    'plot_edges_grid_umap',
    'compare_conditions_edges_umap',
    'compare_neighborhood_enrichment',
    'plot_edges_per_sample',
    'analyze_interaction_drivers',
]