"""
Spatial Edge Visualization Per Sample

Function for visualizing edge connectivity patterns for individual samples
with customizable colors and styles.
"""

from typing import Optional
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from anndata import AnnData

__all__ = [
    "plot_edges_per_sample",
]


def plot_edges_per_sample(
    adata: AnnData,
    edge_matrix_key: str = "edge_conf_matrix_sparse",
    sample_key: str = "sample",
    condition_key: str = "condition",
    max_edges_per_sample: int = 2000,
    edge_color: str = "black",
    edge_linewidth: float = 0.8,
    edge_alpha: float = 0.6,
    cell_color: str = "lightblue",
    cell_size: float = 5,
    cell_alpha: float = 0.5,
    bg_color: str = "white",
    save_prefix: str = "edges_per_sample",
) -> Optional[pd.DataFrame]:
    """
    Plot edges for each sample separately with customizable colors.

    Creates individual plots showing edge connectivity patterns for each sample,
    with edges drawn as lines connecting cells in spatial coordinates.

    Parameters
    ----------
    adata : AnnData
        Annotated data with edge confidence matrix and spatial coordinates.
    edge_matrix_key : str, optional (default: ``"edge_conf_matrix_sparse"``)
        Key for edge confidence matrix in ``adata.uns``.
    sample_key : str, optional (default: ``"sample"``)
        Key in ``adata.obs`` for sample annotations.
    condition_key : str, optional (default: ``"condition"``)
        Key in ``adata.obs`` for condition labels.
    max_edges_per_sample : int, optional (default: 2000)
        Maximum edges to plot per sample (subsampled if exceeded).
    edge_color : str, optional (default: ``'black'``)
        Color for edges.
    edge_linewidth : float, optional (default: 0.8)
        Width of edge lines.
    edge_alpha : float, optional (default: 0.6)
        Transparency of edges (0–1).
    cell_color : str, optional (default: ``'lightblue'``)
        Color for cells.
    cell_size : float, optional (default: 5)
        Size of cell markers.
    cell_alpha : float, optional (default: 0.5)
        Transparency of cells (0–1).
    bg_color : str, optional (default: ``'white'``)
        Background color.
    save_prefix : str, optional (default: ``"edges_per_sample"``)
        Prefix for saved files.

    Returns
    -------
    pd.DataFrame or None
        Statistics for each sample (sample, condition, n_cells, n_edges,
        avg_edges_per_cell).

    Examples
    --------
    >>> import casei
    >>> stats = casei.pl.plot_edges_per_sample(
    ...     adata,
    ...     sample_key="sample",
    ...     edge_color="black",
    ...     cell_color="lightblue",
    ...     edge_linewidth=0.8,
    ...     max_edges_per_sample=2000,
    ... )
    >>> print(stats)
    """
    if edge_matrix_key not in adata.uns:
        raise ValueError(f"'{edge_matrix_key}' not found in adata.uns")

    if "spatial" not in adata.obsm:
        raise ValueError("'spatial' coordinates not found in adata.obsm")

    if sample_key not in adata.obs.columns:
        raise ValueError(f"'{sample_key}' not found in adata.obs")

    edge_matrix = adata.uns[edge_matrix_key]

    print(f"\n{'=' * 70}")
    print(f"PLOTTING EDGES PER SAMPLE")
    print(f"   Edge matrix: {edge_matrix_key}")
    print(f"   Edges: {edge_color}")
    print(f"   Cells: {cell_color}")
    print(f"   Background: {bg_color}")
    print(f"{'=' * 70}")

    samples = sorted(adata.obs[sample_key].unique())
    print(f"\nFound {len(samples)} samples: {samples}")

    sample_stats = []

    for sample in samples:
        adata_sample = adata[adata.obs[sample_key] == sample].copy()
        n_cells = len(adata_sample)
        condition = (
            adata_sample.obs[condition_key].iloc[0]
            if condition_key in adata_sample.obs.columns
            else "Unknown"
        )

        sample_indices = adata.obs.index.get_indexer(adata_sample.obs.index)
        sample_edge_matrix = edge_matrix[sample_indices, :][:, sample_indices]

        edges_coo = sample_edge_matrix.tocoo()
        n_edges_total = len(edges_coo.data)
        n_unique_edges = n_edges_total // 2

        print(f"\nSample: {sample} ({condition})")
        print(f"   Cells: {n_cells}")
        print(f"   Unique edges: {n_unique_edges}")

        if n_unique_edges == 0:
            print(f"   WARNING: No edges found for sample {sample}!")
            continue

        edge_counts = np.array(sample_edge_matrix.sum(axis=1)).flatten()
        avg_edges = edge_counts.mean()

        sample_stats.append(
            {
                "sample": sample,
                "condition": condition,
                "n_cells": n_cells,
                "n_edges": n_unique_edges,
                "avg_edges_per_cell": avg_edges,
            }
        )

        print(f"   Avg edges per cell: {avg_edges:.2f}")

        # Subsample if too many edges
        if n_edges_total > max_edges_per_sample * 2:
            print(f"   Subsampling {max_edges_per_sample} from {n_unique_edges} edges")
            indices = np.random.choice(
                len(edges_coo.data), max_edges_per_sample * 2, replace=False
            )
            edge_rows = edges_coo.row[indices]
            edge_cols = edges_coo.col[indices]
        else:
            edge_rows = edges_coo.row
            edge_cols = edges_coo.col

        spatial_coords = adata_sample.obsm["spatial"]

        fig, ax = plt.subplots(figsize=(14, 12))
        fig.patch.set_facecolor(bg_color)
        ax.set_facecolor(bg_color)

        # Plot edges first (behind cells)
        plotted_edges = 0
        print(f"   Plotting edges...")

        for i, j in zip(edge_rows, edge_cols):
            if i < j:  # Only plot each edge once
                x_coords = [spatial_coords[i, 0], spatial_coords[j, 0]]
                y_coords = [spatial_coords[i, 1], spatial_coords[j, 1]]
                ax.plot(
                    x_coords,
                    y_coords,
                    color=edge_color,
                    alpha=edge_alpha,
                    linewidth=edge_linewidth,
                    zorder=1,
                )
                plotted_edges += 1

        print(f"   Plotted {plotted_edges} edges")

        # Plot cells on top
        ax.scatter(
            spatial_coords[:, 0],
            spatial_coords[:, 1],
            c=cell_color,
            s=cell_size,
            alpha=cell_alpha,
            zorder=2,
            edgecolors="none",
        )

        ax.set_xlabel("Spatial X", fontsize=20, fontweight="bold")
        ax.set_ylabel("Spatial Y", fontsize=20, fontweight="bold")
        ax.set_title(
            f"Sample: {sample} ({condition})\n{n_cells} cells, {plotted_edges} edges",
            fontsize=24,
            fontweight="bold",
        )
        ax.set_aspect("equal")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.tick_params(labelsize=18)

        plt.tight_layout()
        filename = f"{save_prefix}_sample_{sample}.png"
        plt.savefig(filename, dpi=300, bbox_inches="tight", facecolor=bg_color)
        print(f"   Saved: {filename}")
        plt.show()
        plt.close()

    if len(sample_stats) > 0:
        stats_df = pd.DataFrame(sample_stats)
        stats_filename = f"{save_prefix}_statistics.csv"
        stats_df.to_csv(stats_filename, index=False)
        print(f"\nStatistics saved to: {stats_filename}")
        return stats_df
    else:
        print("\nWARNING: No edges found in any sample!")
        return None
