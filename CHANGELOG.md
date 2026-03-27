# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/).

## [0.1.0] — 2025-XX-XX

### Added
- `EdgePredictionMLP` model with symmetric `W @ Wᵀ` parameterization.
- `preprocess_adata` for standard spatial transcriptomics preprocessing.
- `train_edge_predictor` training loop.
- `store_edge_confidence_matrix` for post-training edge storage.
- `contrast_gene_interactions` for differential interaction analysis.
- `plot_edges_per_sample` per-sample spatial edge visualization.
- `analyze_interaction_drivers` interacting-vs-bystander DE pipeline.
- `plot_differential_heatmap`, `plot_gene_network`, `plot_edge_comparison` stubs.
- GitHub Actions CI for Python 3.9–3.12.
