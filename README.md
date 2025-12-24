# Non-Negative Autoencoding for Single-Cell Data

This repository contains the **final training, analysis, and figure-generation code**
used in the Non-negative Autoencoder (NNAE) project, focused on learning
interpretable gene programs from the CELLxGENE Census and benchmarking them
against NMF and other baselines.

The code here supports:

- Training a sparse, tied-weights NNAE on large-scale single-cell data
- Exporting embeddings from the 60M-cell Census
- Downstream statistical analysis and visualization
- Gene set enrichment analysis (GSEA)
- Generation of manuscript and supplemental figures

This repository is intended for **research transparency and collaboration**, not as a
turn-key pipeline.

---

## Repository structure

```text
.
├── code/
│   ├── census_model/        # Model training and embedding export
│   ├── figures/             # Figure-specific analysis and plotting scripts
│   └── gsea/                # Gene set enrichment analysis notebooks
└── figure_output/           # Rendered figures (PDFs)


