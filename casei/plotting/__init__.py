"""
Plotting Functions for Spatial Transcriptomics
===============================================

Visualization tools for spatial edge predictions, gene interaction
networks, and differential analysis results.
"""

from .spatial_edges import plot_edges_per_sample
from .interaction_drivers import analyze_interaction_drivers
from ._core import (
    plot_differential_heatmap,
    plot_gene_network,
    plot_edge_comparison,
)

__all__ = [
    "plot_edges_per_sample",
    "analyze_interaction_drivers",
    "plot_differential_heatmap",
    "plot_gene_network",
    "plot_edge_comparison",
]
