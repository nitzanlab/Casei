"""
Interaction Drivers Analysis

Function for identifying high-confidence interacting cells and their
differential expression signatures.
"""

from typing import Optional, List
import numpy as np
import pandas as pd
import scanpy as sc
import matplotlib.pyplot as plt
from anndata import AnnData

__all__ = [
    "analyze_interaction_drivers",
]


def analyze_interaction_drivers(
    adata: AnnData,
    cell_type_1: str,
    cell_type_2: str,
    condition_key: str = "condition",
    target_cond: str = "aged",
    cell_type_key: str = "celltype",
    matrix_key: str = "edge_conf_matrix_sparse",
    n_top_genes: int = 10,
    save_prefix: Optional[str] = None,
) -> AnnData:
    """
    Identify high-confidence interacting cells and their DE signatures.

    Finds cells of two given types that are connected by predicted edges,
    labels them as *Interacting* vs *Bystander*, and runs Wilcoxon
    differential expression between the two groups.

    Parameters
    ----------
    adata : AnnData
        Annotated data with edge confidence matrix.
    cell_type_1 : str
        First cell type to analyze.
    cell_type_2 : str
        Second cell type to analyze.
    condition_key : str, optional (default: ``"condition"``)
        Key in ``adata.obs`` for condition labels.
    target_cond : str, optional (default: ``"aged"``)
        Condition to restrict the analysis to.
    cell_type_key : str, optional (default: ``"celltype"``)
        Key in ``adata.obs`` for cell-type annotations.
    matrix_key : str, optional (default: ``"edge_conf_matrix_sparse"``)
        Key in ``adata.uns`` for the edge confidence matrix.
    n_top_genes : int, optional (default: 10)
        Number of top DE genes to display.
    save_prefix : str or None, optional
        If provided, dotplots are saved with this prefix.

    Returns
    -------
    AnnData
        The input *adata* with an added ``interaction_status`` column in
        ``adata.obs`` and DE results in ``adata.uns``.

    Examples
    --------
    >>> import casei
    >>> adata = casei.tl.analyze_interaction_drivers(
    ...     adata,
    ...     cell_type_1="T_cell",
    ...     cell_type_2="Macrophage",
    ...     target_cond="aged",
    ... )
    """

    print(f"\n{'=' * 80}")
    print(f"Analyzing Interactions: {cell_type_1} <-> {cell_type_2} ({target_cond})")
    print(f"{'=' * 80}")

    # === 1. VALIDATION & SETUP ===
    if matrix_key not in adata.uns:
        raise KeyError(f"Key '{matrix_key}' not found in adata.uns.")

    adj = adata.uns[matrix_key]

    if "interaction_status" not in adata.obs.columns:
        adata.obs["interaction_status"] = "Other"
    adata.obs["interaction_status"] = adata.obs["interaction_status"].astype(str)

    cond_mask = adata.obs[condition_key] == target_cond

    idx_1 = np.where(cond_mask & (adata.obs[cell_type_key] == cell_type_1))[0]
    idx_2 = np.where(cond_mask & (adata.obs[cell_type_key] == cell_type_2))[0]

    if len(idx_1) == 0:
        print(f"No cells found for {cell_type_1} in {target_cond}.")
        return adata
    if len(idx_2) == 0:
        print(f"No cells found for {cell_type_2} in {target_cond}.")
        return adata

    # === 2. IDENTIFY INTERACTING CELLS ===
    print(f"Tracing high-confidence edges...")

    submatrix = adj[idx_1, :][:, idx_2]

    is_connected_1 = submatrix.getnnz(axis=1) > 0
    is_connected_2 = submatrix.getnnz(axis=0) > 0

    interacting_idx_1 = idx_1[is_connected_1]
    interacting_idx_2 = idx_2[is_connected_2]

    adata.obs.iloc[
        idx_1, adata.obs.columns.get_loc("interaction_status")
    ] = f"{cell_type_1}_Bystander"
    adata.obs.iloc[
        interacting_idx_1, adata.obs.columns.get_loc("interaction_status")
    ] = f"{cell_type_1}_Interacting"

    if cell_type_1 != cell_type_2:
        adata.obs.iloc[
            idx_2, adata.obs.columns.get_loc("interaction_status")
        ] = f"{cell_type_2}_Bystander"
        adata.obs.iloc[
            interacting_idx_2, adata.obs.columns.get_loc("interaction_status")
        ] = f"{cell_type_2}_Interacting"
        types_to_analyze = [cell_type_1, cell_type_2]
    else:
        types_to_analyze = [cell_type_1]

    # === 3. DE & VISUALIZATION LOOP ===
    for ctype in types_to_analyze:
        inter_label = f"{ctype}_Interacting"
        byst_label = f"{ctype}_Bystander"

        n_int = np.sum(adata.obs["interaction_status"] == inter_label)
        n_byst = np.sum(adata.obs["interaction_status"] == byst_label)

        if n_int < 3 or n_byst < 3:
            print(
                f"Skipping DE for {ctype}: Not enough cells (need >3 per group)."
            )
            continue

        print(f"\nPerforming DE for {ctype} ({n_int} vs {n_byst} cells)...")

        key_name = f"rank_genes_{ctype}_interaction"

        try:
            sc.tl.rank_genes_groups(
                adata,
                groupby="interaction_status",
                groups=[inter_label],
                reference=byst_label,
                method="wilcoxon",
                key_added=key_name,
            )
        except Exception as e:
            print(f"  DE failed for {ctype}: {e}")
            continue

        df_res = sc.get.rank_genes_groups_df(adata, group=inter_label, key=key_name)
        top_genes = df_res["names"].head(n_top_genes).tolist()

        print(f"  Top drivers: {', '.join(top_genes[:5])}...")

        # Statistics summary
        print(f"\n  {'─' * 60}")
        print(f"  DE Statistics — {ctype}: Interacting vs Bystander")
        print(f"  {'─' * 60}")
        print(f"  {'Gene':<20} {'Score':>10} {'Log2FC':>10} {'Adj. p-val':>14}")
        print(f"  {'─' * 60}")
        for _, row in df_res.head(n_top_genes).iterrows():
            sig = "*" if row["pvals_adj"] < 0.05 else " "
            print(
                f"  {row['names']:<20}"
                f" {row['scores']:>10.3f}"
                f" {row['logfoldchanges']:>10.3f}"
                f" {row['pvals_adj']:>13.2e}"
                f" {sig}"
            )
        n_sig = (df_res["pvals_adj"] < 0.05).sum()
        print(f"  {'─' * 60}")
        print(f"  Significant genes (adj. p < 0.05): {n_sig} / {len(df_res)}")
        print(f"  {'─' * 60}\n")

        # Dotplot
        print(f"  Generating dotplot...")

        adata_plot = adata[
            adata.obs["interaction_status"].isin([inter_label, byst_label])
        ].copy()

        adata_plot.obs["interaction_status"] = pd.Categorical(
            adata_plot.obs["interaction_status"],
            categories=[inter_label, byst_label],
            ordered=True,
        )

        sc.pl.dotplot(
            adata_plot,
            var_names=top_genes,
            groupby="interaction_status",
            title=f"Drivers of {ctype} Interaction",
            cmap="Reds" if ctype == cell_type_1 else "Blues",
            save=f"_{save_prefix}_{ctype}.png" if save_prefix else None,
            show=True,
        )

        print(f"  Dotplot complete for {ctype}")

    return adata
