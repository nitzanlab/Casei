# Casei

**Condition-Associated Spatial Edge Inference** — spatial transcriptomics edge prediction and differential interaction analysis using bilinear graph classification.

Casei identifies which pairwise relationships between neighboring cells are condition-specific, shifting the unit of biological inference from individual cells to the interactions (edges) between them.

---

## Features

- **Bilinear Edge Prediction** — a symmetric `W @ Wᵀ` architecture that learns condition-specific gene–gene interaction matrices from spatial proximity graphs, grounded in energy-based modeling of pairwise cellular interactions.
- **Training Dynamics Confidence** — edges are ranked by their mean predicted probability across training epochs, capturing continuous biological variation beyond discrete condition labels.
- **Differential Interaction Analysis** — contrast predicted edge confidence between conditions (e.g. young vs. aged, healthy vs. diseased) to find interactions that are gained or lost.
- **Interaction Driver Discovery** — label cells as *Interacting* vs *Bystander* based on predicted edges, then run differential expression to find the genes driving those interactions.
- **Spatial Visualization** — per-sample edge plots, differential heatmaps, gene interaction networks, and scanpy-integrated dotplots.

---

## Installation

### From source (development)

```bash
git clone https://github.com/nitzanlab/Casei.git
cd casei
pip install .
```

### Requirements

- Python ≥ 3.9
- PyTorch ≥ 1.12
- Scanpy ≥ 1.9
- AnnData ≥ 0.9

All dependencies are installed automatically.

---

## Quick Start

```python
import casei

# 1. Set seed for reproducibility
casei.set_random_seed(42)

# 2. Preprocess your AnnData object
adata = casei.preprocess_adata(adata)

# 3. Train the edge prediction model
results = casei.train_edge_predictor(adata, n_epochs=100, hidden_channels=64)

# 4. Store edge confidence matrix (mean predicted probability across training epochs)
adata = casei.store_edge_confidence_matrix(adata, model=results["model"])

# 5. Contrast interactions across conditions — constructs condition-adjusted graphs
adata = casei.contrast_gene_interactions(adata, condition_key="condition")

# 6. Visualize
casei.pl.plot_differential_heatmap(adata)
casei.pl.plot_edges_per_sample(adata, edge_color="black", cell_color="lightblue")
```

---

## Tutorial

A step-by-step tutorial applying Casei to atherosclerosis spatial transcriptomics data (healthy vs. plaque vs. smoker/non-smoker plaques) is available here:

📓 [Smoking & Atherosclerosis Tutorial](https://github.com/nitzanlab/casei_reproducibility/blob/main/smoking_casei_tutorial.ipynb)

---

## Package Structure

```
casei/
├── __init__.py            # Public API
├── _version.py            # Version info
├── models/
│   └── __init__.py        # EdgePredictionMLP
├── plotting/
│   ├── __init__.py        # Plotting API
│   ├── _core.py           # Heatmaps, networks, comparisons
│   ├── spatial_edges.py   # Per-sample edge visualization
│   └── interaction_drivers.py  # Interacting vs Bystander dotplots
└── tools/
    ├── __init__.py        # Tools API
    ├── _preprocessing.py  # preprocess_adata, set_random_seed
    ├── _training.py       # train_edge_predictor
    └── _analysis.py       # contrast_gene_interactions, store_edge_confidence_matrix
```

---

## API Overview

### Models

| Class | Description |
|---|---|
| `casei.EdgePredictionMLP` | Bilinear edge predictor with `W @ Wᵀ` factorization |

### Tools (`casei.tl`)

| Function | Description |
|---|---|
| `preprocess_adata()` | Standard spatial transcriptomics preprocessing |
| `set_random_seed()` | Set all random seeds for reproducibility |
| `train_edge_predictor()` | Train the bilinear edge prediction model |
| `store_edge_confidence_matrix()` | Compute and store per-edge confidence scores |
| `contrast_gene_interactions()` | Build condition-adjusted graphs via differential interaction analysis |

### Plotting (`casei.pl`)

| Function | Description |
|---|---|
| `plot_edges_per_sample()` | Visualize edges on spatial coordinates per sample |
| `plot_differential_heatmap()` | Heatmap of differential interaction scores |
| `plot_gene_network()` | Gene interaction network graph |
| `plot_edge_comparison()` | Compare edge confidence across conditions |
| `analyze_interaction_drivers()` | DE analysis comparing interacting vs. bystander cells |

---

## Analyzing Interaction Drivers

```python
# Identify genes driving T-cell / Macrophage interactions in aged tissue
adata = casei.pl.analyze_interaction_drivers(
    adata,
    cell_type_1="T_cell",
    cell_type_2="Macrophage",
    condition_key="condition",
    target_cond="aged",
    n_top_genes=15,
)
```

This labels cells as `T_cell_Interacting` / `T_cell_Bystander` (and likewise for macrophages), runs Wilcoxon differential expression, prints a summary table, and generates dotplots.

---


## Citation

If you use Casei in your research, please cite:

```bibtex
@article{Karin2026.05.03.722470,
  author = {Karin, Jonathan and Friedman, Roy and Nitzan, Mor},
  title = {Decoding Condition-Specific Cellular Crosstalk in Spatial Omics via Bilinear Edge Classification},
  year = {2026},
  doi = {10.64898/2026.05.03.722470},
  publisher = {bioRxiv},
  journal = {bioRxiv}
}
```

[→ View preprint](https://doi.org/10.64898/2026.05.03.722470)

---

## License

MIT — see [LICENSE](LICENSE) for details.
