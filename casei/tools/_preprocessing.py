"""
Preprocessing utilities for spatial transcriptomics data.
"""

import random
import numpy as np
from anndata import AnnData

__all__ = ["preprocess_adata", "set_random_seed"]


def set_random_seed(seed: int = 42) -> None:
    """
    Set random seeds for reproducibility across numpy and random.

    Parameters
    ----------
    seed : int, optional (default: 42)
        Random seed value.
    """
    random.seed(seed)
    np.random.seed(seed)

    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def preprocess_adata(
    adata: AnnData,
    min_genes: int = 200,
    min_cells: int = 3,
    n_top_genes: int = 2000,
    normalize_total: bool = True,
    log_transform: bool = True,
) -> AnnData:
    """
    Standard preprocessing pipeline for spatial transcriptomics data.

    Parameters
    ----------
    adata : AnnData
        Raw annotated data matrix.
    min_genes : int, optional (default: 200)
        Minimum number of genes per cell.
    min_cells : int, optional (default: 3)
        Minimum number of cells per gene.
    n_top_genes : int, optional (default: 2000)
        Number of highly variable genes to select.
    normalize_total : bool, optional (default: True)
        Whether to normalize total counts per cell.
    log_transform : bool, optional (default: True)
        Whether to log-transform the data.

    Returns
    -------
    AnnData
        Preprocessed annotated data.
    """
    raise NotImplementedError(
        "preprocess_adata is a placeholder. Implement your preprocessing pipeline here."
    )
