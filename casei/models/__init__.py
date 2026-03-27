"""
Neural Network Models for Edge Prediction
==========================================

This module contains graph neural network architectures for predicting
edges in spatial transcriptomics data.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


class EdgePredictionMLP(nn.Module):
    """
    Edge prediction model using W @ W^T parameterization.

    This architecture ensures the learned matrix is symmetric and positive
    semi-definite by factorizing it as W @ W^T where W has shape
    (classes, genes, rank).

    Parameters
    ----------
    in_channels : int
        Number of input features (genes)
    hidden_channels : int, optional (default: 64)
        Rank of the factorization. Lower values create more constrained models.
    out_channels : int, optional (default: 2)
        Number of output classes/conditions

    Attributes
    ----------
    W : torch.nn.Parameter
        Learnable weight matrix of shape (classes, genes, rank)
    rank : int
        Rank of the factorization

    Examples
    --------
    >>> import casei
    >>> model = casei.models.EdgePredictionMLP(
    ...     in_channels=2000,
    ...     hidden_channels=64,
    ...     out_channels=2
    ... )
    >>> # Forward pass
    >>> x = torch.randn(100, 2000)  # 100 cells, 2000 genes
    >>> edge_index = torch.randint(0, 100, (2, 500))  # 500 edges
    >>> out = model(x, edge_index)
    >>> out.shape
    torch.Size([500, 2])
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int = 64,
        out_channels: int = 2,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.rank = hidden_channels

        # Learn W of shape (classes, genes, rank)
        # The effective weight matrix is W @ W^T which is (classes, genes, genes)
        self.W = nn.Parameter(
            torch.randn(out_channels, in_channels, hidden_channels)
        )
        # Initialize with small values for stability
        nn.init.normal_(self.W, mean=0, std=0.001)

    def get_active_weights(self) -> torch.Tensor:
        """
        Get the effective weight matrix W @ W^T.

        Returns
        -------
        torch.Tensor
            Weight tensor of shape (classes, genes, genes)
        """
        WT = self.W.transpose(1, 2)  # (classes, rank, genes)
        W_WT = torch.bmm(self.W, WT)  # (classes, genes, genes)
        return W_WT

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute edge predictions.

        Efficiently computes x_i^T @ W @ W^T @ x_j without forming full W @ W^T.

        Parameters
        ----------
        x : torch.Tensor
            Node features of shape (n_nodes, n_genes)
        edge_index : torch.Tensor
            Edge indices of shape (2, n_edges)

        Returns
        -------
        torch.Tensor
            Edge predictions of shape (n_edges, n_classes)
        """
        x_i = x[edge_index[0]]  # (edges, genes)
        x_j = x[edge_index[1]]  # (edges, genes)

        # Efficient computation: x_i^T @ W @ (W^T @ x_j)
        v_j = torch.einsum("cgr,eg->ecr", self.W, x_j)
        bilinear_out = torch.einsum("eg,cgr,ecr->ec", x_i, self.W, v_j)

        return bilinear_out

    def get_regularization_loss(
        self,
        lambda_l1: float = 1e-5,
        lambda_l2: float = 1e-4,
    ) -> torch.Tensor:
        """
        Compute regularization loss on W.

        Parameters
        ----------
        lambda_l1 : float, optional (default: 1e-5)
            L1 penalty coefficient for sparsity
        lambda_l2 : float, optional (default: 1e-4)
            L2 penalty coefficient for weight decay

        Returns
        -------
        torch.Tensor
            Combined regularization loss
        """
        l1_loss = torch.sum(torch.abs(self.W))
        l2_loss = torch.sum(self.W**2)
        return (lambda_l1 * l1_loss) + (lambda_l2 * l2_loss)

    @property
    def bilinear(self):
        """
        Helper property for visualization.

        Returns a mock bilinear object with the effective weight matrix.
        """

        class MockBilinear:
            def __init__(self, weight):
                self.weight = weight

        with torch.no_grad():
            return MockBilinear(self.get_active_weights())


__all__ = ["EdgePredictionMLP"]
