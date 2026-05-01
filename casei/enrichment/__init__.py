"""
Differential Interaction Program Enrichment Analysis
====================================================

This module provides functions for eigendecomposition-based analysis of
differential gene-gene interaction matrices and GO/pathway enrichment.
Developed as part of the Casei framework for spatial omics analysis.
"""

from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import linalg
import torch
from anndata import AnnData

__all__ = [
    'decompose_differential_matrix',
    'run_enrichment_analysis',
    'visualize_eigenvalue_spectrum',
    'compare_program_overlaps',
    'save_program_summary',
    'contrast_gene_gene_interactions',
    'analyze_differential_interaction_programs_with_enrichment',
]


def contrast_gene_gene_interactions(
    model,
    adata_filtered: AnnData,
    le,
    cond1: str,
    cond2: str,
    top_k: int = 20,
    min_abs_weight: float = 0.0
) -> Tuple[pd.DataFrame, pd.DataFrame, np.ndarray]:
    """
    Compute differential gene-gene interactions: (W⊗W^T)_cond1 - (W⊗W^T)_cond2

    Parameters
    ----------
    model : EdgePredictionMLP
        Trained model
    adata_filtered : AnnData
        Filtered AnnData
    le : LabelEncoder
        Label encoder for conditions
    cond1 : str
        First condition (e.g., "healthy")
    cond2 : str or list
        Second condition(s) (e.g., "diseased")
    top_k : int
        Number of top interactions to return
    min_abs_weight : float
        Minimum absolute weight threshold

    Returns
    -------
    pos_df : pd.DataFrame
        Positive interactions (enriched in cond1)
    neg_df : pd.DataFrame
        Negative interactions (enriched in cond2)
    delta : np.ndarray
        Differential matrix
    """
    # Get condition indices
    c1 = le.transform([cond1])[0]

    # Handle cond2 as single condition or list
    if isinstance(cond2, list):
        c2_indices = [le.transform([c])[0] for c in cond2]
    else:
        c2_indices = [le.transform([cond2])[0]]

    # Get W W^T matrices (low-rank factorization of interactions)
    with torch.no_grad():
        WWt = model.get_active_weights().cpu().numpy()

    M1 = WWt[c1]

    # Average over cond2 if multiple conditions are provided
    if len(c2_indices) > 1:
        M2 = np.mean([WWt[idx] for idx in c2_indices], axis=0)
    else:
        M2 = WWt[c2_indices[0]]

    # Compute contrast
    delta = M1 - M2   # (genes x genes)
    gene_names = adata_filtered.var_names.to_numpy()

    # Remove diagonal to focus on gene-gene interactions
    delta_no_diag = delta.copy()
    np.fill_diagonal(delta_no_diag, 0)

    # Helper to extract top interactions
    def extract_top(mat, sign=1):
        values = sign * mat
        flat_idx = np.argsort(values.ravel())[::-1]
        seen = set()
        results = []

        for idx in flat_idx:
            i, j = np.unravel_index(idx, mat.shape)
            if i >= j:  # Only take upper triangle
                continue

            val = mat[i, j]
            if sign * val <= min_abs_weight:
                break

            pair = (i, j)
            if pair in seen:
                continue

            seen.add(pair)
            results.append({
                "gene_i": gene_names[i],
                "gene_j": gene_names[j],
                "delta_weight": val,
                "abs_delta": abs(val)
            })

            if len(results) >= top_k:
                break

        return pd.DataFrame(results)

    # Extract results
    pos_df = extract_top(delta_no_diag, sign=+1)
    neg_df = extract_top(delta_no_diag, sign=-1)

    return pos_df, neg_df, delta


def decompose_differential_matrix(delta_matrix: np.ndarray) -> Dict:
    """
    Decompose differential matrix into positive and negative components.

    Performs eigendecomposition on Δ+ (interactions enriched in condition 1)
    and Δ− (interactions enriched in condition 2).

    Parameters
    ----------
    delta_matrix : np.ndarray
        (genes × genes) differential interaction matrix

    Returns
    -------
    dict
        Dictionary containing eigenvalues, eigenvectors, and component matrices.

    Examples
    --------
    >>> import casei as ci
    >>> pos_df, neg_df, delta = ci.enr.contrast_gene_gene_interactions(...)
    >>> decomp = ci.enr.decompose_differential_matrix(delta)
    """
    # Split into positive and negative parts
    delta_pos = np.maximum(delta_matrix, 0)
    delta_neg = np.maximum(-delta_matrix, 0)

    # Ensure symmetry for numerical stability
    delta_pos = (delta_pos + delta_pos.T) / 2
    delta_neg = (delta_neg + delta_neg.T) / 2

    # Eigendecomposition
    eigvals_pos, eigvecs_pos = linalg.eigh(delta_pos)
    eigvals_neg, eigvecs_neg = linalg.eigh(delta_neg)

    # Sort by magnitude (descending) to identify top programs
    idx_pos = np.argsort(np.abs(eigvals_pos))[::-1]
    idx_neg = np.argsort(np.abs(eigvals_neg))[::-1]

    return {
        "pos_eigvals": eigvals_pos[idx_pos],
        "pos_eigvecs": eigvecs_pos[:, idx_pos],
        "neg_eigvals": eigvals_neg[idx_neg],
        "neg_eigvecs": eigvecs_neg[:, idx_neg],
        "delta_pos": delta_pos,
        "delta_neg": delta_neg
    }


def analyze_differential_interaction_programs_with_enrichment(
    model,
    adata_filtered: AnnData,
    le,
    cond1: str,
    cond2: str,
    n_programs: int = 3,
    top_n_genes: int = 50,
    organism: str = "mouse",
    gene_sets: List[str] = None,
    save_prefix: str = "interaction_programs"
) -> Tuple[Dict, Dict]:
    """
    Complete enrichment analysis pipeline for differential gene-gene interactions.
    """
    if gene_sets is None:
        gene_sets = ["GO_Biological_Process_2023", "KEGG_2021_Human"]

    try:
        import gseapy as gp
    except ImportError:
        raise ImportError("gseapy is required. Install it with: pip install gseapy")

    gene_names = adata_filtered.var_names.to_numpy()

    print(f"\n{'='*70}")
    print(f"🔬 DIFFERENTIAL INTERACTION PROGRAM ANALYSIS: {cond1} vs. {cond2}")
    print(f"{'='*70}")

    pos_df, neg_df, delta_matrix = contrast_gene_gene_interactions(
        model=model, adata_filtered=adata_filtered, le=le,
        cond1=cond1, cond2=cond2, top_k=50
    )

    decomp_results = decompose_differential_matrix(delta_matrix)
    enrichment_results = {}

    def run_for_side(eigvecs, eigvals, label, color, n_programs):
        side_results = []
        for k in range(min(n_programs, eigvecs.shape[1])):
            v = eigvecs[:, k]
            lam = eigvals[k]

            print(f"\n📊 Program {k+1} - Enriched in {label} (λ={lam:.4e})")

            # Rank genes by absolute loading
            idx = np.argsort(np.abs(v))[::-1][:top_n_genes]
            top_genes = [gene_names[i] for i in idx]
            weights = v[idx]
            top_genes_upper = [g.upper() for g in top_genes]

            # --- Gene Loading Bar Plot ---
            fig, ax = plt.subplots(figsize=(8, 9))
            n_show = min(20, len(top_genes))
            colors_bar = [color if w > 0 else 'lightgray' for w in weights[:n_show]]
            ax.barh(range(n_show), weights[:n_show][::-1], color=colors_bar[::-1])
            ax.set_yticks(range(n_show))
            ax.set_yticklabels(top_genes[:n_show][::-1], fontsize=15)
            ax.set_xlabel("Gene Loading", fontsize=18, fontweight='bold')
            ax.set_title(f"Interaction Program {k+1}\nEnriched in {label}\nλ = {lam:.2e}",
                         fontweight="bold", fontsize=24)
            plt.tight_layout()
            plt.savefig(f"{save_prefix}_{label}_program_{k+1}_genes.png", dpi=300)
            plt.show(); plt.close()

            # --- GO Enrichment ---
            enrichment_data = {}
            for gene_set in gene_sets:
                try:
                    enr = gp.enrichr(gene_list=top_genes_upper, gene_sets=[gene_set],
                                     organism=organism, outdir=None, cutoff=0.5)

                    if enr.results is not None and len(enr.results) > 0:
                        sig_results = enr.results[enr.results['Adjusted P-value'] < 0.05]
                        if len(sig_results) > 0:
                            # Save and Plot
                            csv_fn = f"{save_prefix}_{label}_program_{k+1}_{gene_set}.csv"
                            sig_results.to_csv(csv_fn, index=False)

                            fig, ax = plt.subplots(figsize=(14, 8))
                            plot_df = sig_results.head(10).copy().sort_values('Adjusted P-value', ascending=False)
                            ax.barh(np.arange(len(plot_df)), -np.log10(plot_df['Adjusted P-value']), color=color)
                            ax.set_yticks(np.arange(len(plot_df)))
                            ax.set_yticklabels(plot_df['Term'], fontsize=18)
                            ax.set_xlabel('-log10(Adjusted P-value)', fontsize=18, fontweight='bold')
                            ax.set_title(f"{gene_set}\nProgram {k+1} ({label})", fontsize=20, fontweight='bold')
                            plt.tight_layout()
                            plt.savefig(f"{save_prefix}_{label}_program_{k+1}_{gene_set}_barplot.png", dpi=300)
                            plt.show(); plt.close()
                            enrichment_data[gene_set] = sig_results
                except Exception as e:
                    print(f"❌ Enrichment failed: {e}")
                    enrichment_data[gene_set] = None

            side_results.append({'program_id': k+1, 'eigenvalue': lam, 'top_genes': top_genes, 'enrichment': enrichment_data})
        return side_results

    enrichment_results['positive'] = run_for_side(decomp_results["pos_eigvecs"], decomp_results["pos_eigvals"], cond1, "salmon", n_programs)
    enrichment_results['negative'] = run_for_side(decomp_results["neg_eigvecs"], decomp_results["neg_eigvals"], cond2, "steelblue", n_programs)

    return decomp_results, enrichment_results


def visualize_eigenvalue_spectrum(
    decomp_results: Dict,
    cond1: str,
    cond2: str,
    save_prefix: str = "eigenvalue_spectrum"
) -> None:
    """Plot eigenvalue spectrum for positive and negative components."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7))

    for ax, evals, label, color in zip([ax1, ax2],
                                      [decomp_results['pos_eigvals'], decomp_results['neg_eigvals']],
                                      [cond1, cond2], ['salmon', 'steelblue']):
        ax.plot(range(1, len(evals)+1), evals, 'o-', color=color, linewidth=3, markersize=10)
        ax.set_xlabel('Program Index', fontsize=18, fontweight='bold')
        ax.set_ylabel('Eigenvalue', fontsize=18, fontweight='bold')
        ax.set_title(f'Eigenvalue Spectrum: {label}', fontsize=20, fontweight='bold')
        ax.set_xlim(0, 50); ax.grid(alpha=0.3); ax.tick_params(labelsize=15)

    plt.tight_layout()
    plt.savefig(f"{save_prefix}_{cond1}_vs_{cond2}.png", dpi=300)
    plt.show(); plt.close()


def compare_program_overlaps(enrichment_results: Dict, cond1: str, cond2: str, top_n: int = 20) -> None:
    """Compare gene overlaps between programs from different conditions."""
    print(f"\n🔄 GENE OVERLAP BETWEEN PROGRAMS ({cond1} vs {cond2})")
    for i, pos_prog in enumerate(enrichment_results['positive']):
        pos_genes = set(pos_prog['top_genes'][:top_n])
        for j, neg_prog in enumerate(enrichment_results['negative']):
            neg_genes = set(neg_prog['top_genes'][:top_n])
            overlap = pos_genes & neg_genes
            if overlap:
                jaccard = len(overlap) / len(pos_genes | neg_genes)
                print(f"{cond1} P{i+1} ↔ {cond2} P{j+1}: {len(overlap)}/{top_n} genes (Jaccard: {jaccard:.3f})")


def save_program_summary(enrichment_results: Dict, cond1: str, cond2: str, save_prefix: str = "program_summary") -> pd.DataFrame:
    """Save comprehensive summary of all programs to CSV."""
    summary = []
    for side, label in [('positive', cond1), ('negative', cond2)]:
        for prog in enrichment_results[side]:
            summary.append({
                'condition': label, 'program_id': prog['program_id'], 'eigenvalue': prog['eigenvalue'],
                'top_genes': ', '.join(prog['top_genes'][:20]),
                'n_enriched_terms': sum(len(df) if df is not None else 0 for df in prog['enrichment'].values())
            })
    summary_df = pd.DataFrame(summary)
    summary_df.to_csv(f"{save_prefix}_{cond1}_vs_{cond2}.csv", index=False)
    return summary_df


def run_enrichment_analysis(
    model,
    adata_filtered: AnnData,
    le,
    cond1: str,
    cond2: str,
    n_programs: int = 3,
    top_n_genes: int = 50,
    organism: str = "mouse",
    gene_sets: List[str] = None,
    save_prefix: str = "interaction_programs"
) -> Tuple[Dict, Dict, pd.DataFrame]:
    """
    Main entry point: Run complete differential interaction program analysis with enrichment.
    """
    decomp, enrich = analyze_differential_interaction_programs_with_enrichment(
        model, adata_filtered, le, cond1, cond2, n_programs, top_n_genes, organism, gene_sets, save_prefix
    )
    visualize_eigenvalue_spectrum(decomp, cond1, cond2, f"{save_prefix}_eigenspectrum")
    compare_program_overlaps(enrich, cond1, cond2, top_n=20)
    summary_df = save_program_summary(enrich, cond1, cond2, f"{save_prefix}_summary")
    return decomp, enrich, summary_df