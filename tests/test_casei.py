"""Tests for casei package."""

import pytest
import torch
import numpy as np


class TestEdgePredictionMLP:
    """Tests for the EdgePredictionMLP model."""

    def test_init(self):
        from casei.models import EdgePredictionMLP

        model = EdgePredictionMLP(in_channels=100, hidden_channels=16, out_channels=2)
        assert model.in_channels == 100
        assert model.rank == 16
        assert model.out_channels == 2
        assert model.W.shape == (2, 100, 16)

    def test_forward_shape(self):
        from casei.models import EdgePredictionMLP

        model = EdgePredictionMLP(in_channels=50, hidden_channels=8, out_channels=3)
        x = torch.randn(20, 50)
        edge_index = torch.randint(0, 20, (2, 40))
        out = model(x, edge_index)
        assert out.shape == (40, 3)

    def test_get_active_weights(self):
        from casei.models import EdgePredictionMLP

        model = EdgePredictionMLP(in_channels=30, hidden_channels=10, out_channels=2)
        W_WT = model.get_active_weights()
        assert W_WT.shape == (2, 30, 30)

        # W @ W^T must be symmetric
        diff = (W_WT - W_WT.transpose(1, 2)).abs().max().item()
        assert diff < 1e-5, "W @ W^T should be symmetric"

    def test_regularization_loss(self):
        from casei.models import EdgePredictionMLP

        model = EdgePredictionMLP(in_channels=20, hidden_channels=4, out_channels=2)
        loss = model.get_regularization_loss(lambda_l1=1e-5, lambda_l2=1e-4)
        assert loss.ndim == 0  # scalar
        assert loss.item() >= 0

    def test_bilinear_property(self):
        from casei.models import EdgePredictionMLP

        model = EdgePredictionMLP(in_channels=10, hidden_channels=4, out_channels=2)
        mock = model.bilinear
        assert hasattr(mock, "weight")
        assert mock.weight.shape == (2, 10, 10)


class TestVersion:
    """Test version info."""

    def test_version_string(self):
        import casei

        assert isinstance(casei.__version__, str)
        parts = casei.__version__.split(".")
        assert len(parts) == 3

    def test_version_info_tuple(self):
        from casei._version import __version_info__

        assert isinstance(__version_info__, tuple)
        assert all(isinstance(v, int) for v in __version_info__)


class TestImports:
    """Test that the public API is importable."""

    def test_top_level_imports(self):
        import casei

        assert hasattr(casei, "models")
        assert hasattr(casei, "pl")
        assert hasattr(casei, "tl")
        assert hasattr(casei, "EdgePredictionMLP")

    def test_plotting_imports(self):
        from casei.plotting import plot_edges_per_sample
        from casei.plotting import analyze_interaction_drivers

        assert callable(plot_edges_per_sample)
        assert callable(analyze_interaction_drivers)

    def test_tools_imports(self):
        from casei.tools import set_random_seed

        assert callable(set_random_seed)


class TestSetRandomSeed:
    """Test reproducibility utility."""

    def test_numpy_seed(self):
        from casei.tools import set_random_seed

        set_random_seed(0)
        a = np.random.rand(5)
        set_random_seed(0)
        b = np.random.rand(5)
        np.testing.assert_array_equal(a, b)

    def test_torch_seed(self):
        from casei.tools import set_random_seed

        set_random_seed(0)
        a = torch.randn(5)
        set_random_seed(0)
        b = torch.randn(5)
        assert torch.equal(a, b)
