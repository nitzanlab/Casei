"""
Core Plotting Functions
=======================

Visualization utilities for differential interaction heatmaps,
gene networks, and edge comparison plots.
"""

from typing import Optional, List, Dict
import numpy as np
import matplotlib.pyplot as plt
from anndata import AnnData

__all__ = [
    "plot_differential_heatmap",
    "plot_gene_network",
    "plot_edge_comparison",
]


def plot_differential_heatmap(
    adata: AnnData,
    diff_key: str = "differential_interactions",
    top_n: int = 50,
    cmap: str = "RdBu_r",
    figsize: tuple = (12, 10),
    save: Optional[str] = None,
    **kwargs,
) -> None:
    """
    Plot a heatmap of differential gene–gene interaction scores.

    Parameters
    ----------
    adata : AnnData
        Annotated data containing differential interaction results.
    diff_key : str, optional
        Key in ``adata.uns`` storing the differential matrix.
    top_n : int, optional
        Number of top interactions to display.
    cmap : str, optional
        Matplotlib colormap name.
    figsize : tuple, optional
        Figure size in inches.
    save : str or None, optional
        Path to save the figure.  ``None`` displays interactively.
    **kwargs
        Additional keyword arguments passed to ``matplotlib.pyplot.imshow``.
    """
    raise NotImplementedError(
        "plot_differential_heatmap is a placeholder. "
        "Implement your heatmap logic here."
    )


def plot_gene_network(
    adata: AnnData,
    gene_list: Optional[List[str]] = None,
    threshold: float = 0.5,
    layout: str = "spring",
    figsize: tuple = (10, 10),
    save: Optional[str] = None,
    **kwargs,
) -> None:
    """
    Plot a gene interaction network graph.

    Parameters
    ----------
    adata : AnnData
        Annotated data containing edge predictions.
    gene_list : list of str or None, optional
        Genes to include. ``None`` uses all genes.
    threshold : float, optional
        Minimum edge confidence to draw.
    layout : str, optional
        Graph layout algorithm (``"spring"``, ``"circular"``, etc.).
    figsize : tuple, optional
        Figure size in inches.
    save : str or None, optional
        Path to save the figure.
    **kwargs
        Additional keyword arguments passed to ``networkx.draw``.
    """
    raise NotImplementedError(
        "plot_gene_network is a placeholder. "
        "Implement your network visualization here."
    )


def plot_edge_comparison(
    adata: AnnData,
    conditions: Optional[List[str]] = None,
    figsize: tuple = (14, 6),
    save: Optional[str] = None,
    **kwargs,
) -> None:
    """
    Compare predicted edges across conditions.

    Parameters
    ----------
    adata : AnnData
        Annotated data containing per-condition edge predictions.
    conditions : list of str or None, optional
        Conditions to compare.  ``None`` uses all available.
    figsize : tuple, optional
        Figure size in inches.
    save : str or None, optional
        Path to save the figure.
    **kwargs
        Additional keyword arguments.
    """
    raise NotImplementedError(
        "plot_edge_comparison is a placeholder. "
        "Implement your comparison logic here."
    )
