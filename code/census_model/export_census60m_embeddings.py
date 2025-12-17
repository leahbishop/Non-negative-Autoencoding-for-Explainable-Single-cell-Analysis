# IMPORT LIBRARIES

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader 
import pandas as pd
import scipy.sparse as sp
from glob import glob
import numpy as np


# CONSTRUCT TIED-WEIGHTS AUTOENCODER 

# Single-layer autoencoder with tied weights
# Architecture: Input (x) → Linear (W) → Hidden (256) → Linear (Wᵗ) → Output (x̂)

# Define autoencoder class with tied weights
class Autoencoder(nn.Module):
    def __init__(self, input_dim, latent_dim=256):
        super(Autoencoder, self).__init__()

        # Encoder: linear projection to latent space (no bias, no activation)
        self.encoder = nn.Linear(input_dim, latent_dim, bias=False)

    def forward(self, x):
        # Project input to latent space
        h = self.encoder(x) #  H = X W^T

        # Reconstruct input from latent space using tied weights (h @ Wᵗ)
        x_hat = torch.matmul(h, self.encoder.weight) # X̂ = H W
        return x_hat, h
    

# LOAD EXPRESSION MATRIX AND METADATA

# Load gene metadata (used for all chunks)
gene_metadata = pd.read_pickle("/mnt/projects/debruinz_project/bisholea/capstone/60M_human_gene_metadata.pkl")
print("Gene metadata shape:", gene_metadata.shape, flush=True)
print("Gene metadata columns:", gene_metadata.columns, flush=True)

# Prepare chunk file lists
count_files = sorted(glob("/mnt/projects/debruinz_project/bisholea/capstone/60M_human_counts/human_counts_*.npz"))
meta_files  = sorted(glob("/mnt/projects/debruinz_project/bisholea/capstone/60M_human_metadata/human_metadata_*.pkl"))
assert len(count_files) == len(meta_files), "Mismatched count and metadata files"

# REBUILD THE MODEL WITH INITIALIZED WEIGHTS

# Rebuild the model
model = Autoencoder(input_dim=gene_metadata.shape[0], latent_dim=256)

# Load trained weights
model.load_state_dict(torch.load("/mnt/projects/debruinz_project/bisholea/capstone/60M Model/60M_model_state_dict_256.pth", map_location=torch.device("cpu")))
model.eval()

# LOAD ALL CHUNKS FOR ANALYSIS (Full analysis, saving embeddings, etc.)

all_embeddings = []
all_metadata = []

batch_size = 512

class SparseDataset(Dataset):
    def __init__(self, X_sparse):
        self.X = X_sparse
    def __len__(self):
        return self.X.shape[0]
    def __getitem__(self, idx):
        x = self.X[idx].toarray().squeeze()
        return torch.tensor(x, dtype=torch.float32)


for count_path, meta_path in zip(count_files, meta_files):
    print(f"Processing {count_path} ...", flush=True)

    X_sparse = sp.load_npz(count_path)
    metadata = pd.read_pickle(meta_path)

    # Filter to "normal"
    mask = metadata["disease"] == "normal"
    X_sparse = X_sparse[mask.values]
    metadata = metadata[mask].reset_index(drop=True)

    dataset = SparseDataset(X_sparse)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    chunk_embeddings = []

    for x_batch in loader:
        with torch.no_grad():
            _, h_batch = model(x_batch)
        chunk_embeddings.append(h_batch.cpu().numpy())

    # Stack embeddings from the current chunk
    chunk_H = np.vstack(chunk_embeddings)
    all_embeddings.append(chunk_H)
    all_metadata.append(metadata)

# Combine all
H_all = np.vstack(all_embeddings)
metadata_all = pd.concat(all_metadata, ignore_index=True)

print("Shape of metadata_all:", metadata_all.shape)
assert metadata_all.shape[0] == H_all.shape[0], "Mismatch between embeddings and metadata rows!"

# SAVE CELL EMBEDDINGS, WEIGHT MATRIX, AND GENE NAMES

# Save cell embeddings (already computed)
print("Saving cell embeddings...", flush=True)
print("H_all shape:", H_all.shape, flush=True)  # (cells x latent_dim)
np.save("/mnt/projects/debruinz_project/bisholea/capstone/60M Model/cell_embeddings_60M_256.npy", H_all)

# Save metadata
metadata_all.to_pickle("/mnt/projects/debruinz_project/bisholea/capstone/cell_metadata_60M.pkl")

# Extract and filter the weight matrix
print("Extracting and filtering weight matrix...", flush=True)
W_raw = model.encoder.weight.detach().cpu().numpy()  # (latent_dim x genes)
print("Raw weight matrix shape (latent_dim x genes):", W_raw.shape, flush=True)

W = W_raw.T  # (genes x latent_dim)
print("Transposed weight matrix shape (genes x latent_dim):", W.shape, flush=True)

# Filter gene names (remove Ensembl IDs)
gene_names = gene_metadata["feature_name"].tolist()
gene_names_np = np.array(gene_names, dtype=object)
mask = np.char.find(gene_names_np.astype(str), 'ENSG') == -1
W_filtered = W[mask]
filtered_gene_names = gene_names_np[mask]

print("Original number of genes:", len(gene_names), flush=True)
print("Filtered gene count (non-Ensembl):", len(filtered_gene_names), flush=True)
print("Filtered weight matrix shape:", W_filtered.shape, flush=True)  # (filtered_genes x latent_dim)

# Save filtered weights and gene names
print("Saving filtered weight matrix and gene names...", flush=True)
np.save("/mnt/projects/debruinz_project/bisholea/capstone/encoder_weight_matrix_filtered_60M.npy", W_filtered)
np.save("/mnt/projects/debruinz_project/bisholea/capstone/gene_names_filtered_60M.npy", filtered_gene_names)


print("All outputs saved successfully.", flush=True)

