"""
Analysis utilities for edge confidence and differential interactions.
"""

from typing import Optional, List
from anndata import AnnData

__all__ = ["contrast_gene_interactions", "store_edge_confidence_matrix"]


def store_edge_confidence_matrix(
    adata: AnnData,
    model=None,
    key: str = "edge_conf_matrix_sparse",
    threshold: float = 0.5,
) -> AnnData:
    """
    Compute and store the edge confidence matrix in ``adata.uns``.

    Parameters
    ----------
    adata : AnnData
        Annotated data with spatial coordinates.
    model : EdgePredictionMLP or None
        Trained model. If ``None``, uses the model stored in ``adata.uns``.
    key : str, optional (default: ``"edge_conf_matrix_sparse"``)
        Key under which to store the matrix in ``adata.uns``.
    threshold : float, optional (default: 0.5)
        Confidence threshold for keeping edges.

    Returns
    -------
    AnnData
        The input *adata* with the confidence matrix stored in ``adata.uns[key]``.
    """
    raise NotImplementedError(
        "store_edge_confidence_matrix is a placeholder. Implement here."
    )


def contrast_gene_interactions(
    adata: AnnData,
    condition_key: str = "condition",
    conditions: Optional[List[str]] = None,
    key_added: str = "differential_interactions",
) -> AnnData:
    """
    Compute differential gene–gene interaction scores across conditions.

    Parameters
    ----------
    adata : AnnData
        Annotated data with per-condition edge predictions.
    condition_key : str, optional (default: ``"condition"``)
        Column in ``adata.obs`` specifying condition labels.
    conditions : list of str or None, optional
        Two conditions to contrast. ``None`` uses the first two unique values.
    key_added : str, optional (default: ``"differential_interactions"``)
        Key in ``adata.uns`` to store results.

    Returns
    -------
    AnnData
        The input *adata* with differential results in ``adata.uns[key_added]``.
    """
    raise NotImplementedError(
        "contrast_gene_interactions is a placeholder. Implement here."
    )
