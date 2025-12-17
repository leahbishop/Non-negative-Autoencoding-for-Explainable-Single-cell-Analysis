# IMPORT LIBRARIES

print("Starting script...", flush=True)

import scipy.sparse as sp
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim
import pickle
from glob import glob
from torch.utils.data import Dataset
import time

class SparseDataset(Dataset):
    def __init__(self, X_sparse):
        self.X = X_sparse

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return idx  # just return the row index
    
# CONSTRUCT TIED-WEIGHTS AUTOENCODER AND INITIALIZE WEIGHTS (cleaned)

# Single-layer autoencoder with tied weights
# Architecture: Input (x) → Linear (W) → Hidden (128D) → Linear (Wᵗ) → Output (x̂)

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
    
# Initialize encoder weights with strictly positive, ~orthogonal values
# Minimizes ‖W Wᵀ – I‖² using iterative updates with non-negativity constraint

# Define initialization function for strictly positive and ~orthogonal weights and set parameters
def init_positive_orthogonal_weights(
    weight: nn.Parameter,
    max_iter: int = 2000,
    tol: float = 1e-3,          
    step_size: float = 0.004,
    patience: int = 25,        
    eps_change: float = 1e-4,  
    verbose: bool = True,
):

    with torch.no_grad():
        # Start with small strictly positive values
        W   = torch.rand_like(weight) * 0.01 + 1e-4
        I   = torch.eye(W.size(0), device=W.device)
        log = []

        best_loss = float("inf")
        stall     = 0

        for itr in range(max_iter):
            # Compute orthogonality loss: ‖W Wᵀ – I‖²
            diff = W @ W.T - I
            loss = (diff ** 2).sum().item()
            log.append(loss)

            if verbose and itr % 25 == 0:
                print(f"[init] iter {itr:4d}  ‖W Wᵀ – I‖² = {loss:,.4f}", flush=True)

            # Stop initialization if loss is below tolerance
            if loss < tol:
                if verbose:
                    print(f"[init] reached tol={tol} at iter {itr}", flush=True)
                break

            # Early stop if loss plateaus
            if best_loss - loss < eps_change:
                stall += 1
            else:
                best_loss = loss
                stall     = 0
            if stall >= patience:
                if verbose:
                    print(f"[init] early stop at iter {itr} "
                          f"(no Δ>{eps_change} for {patience} iters)", flush=True)
                break

            # Gradient-like update and clamp to keep weights non-negative
            W = W - step_size * (diff @ W)
            W.clamp_(min=0.0)   

        # Copy initialized weights back to model and print summary
        weight.data.copy_(W)
        if verbose:
            print(f"[init] DONE after {itr+1} iters  "
                  f"final ‖W Wᵀ – I‖² = {log[-1]:,.4f}", flush=True)

        return log
    
# LOAD EXPRESSION MATRIX AND METADATA

# Load gene metadata (used for all chunks)
gene_metadata = pd.read_pickle("/mnt/projects/debruinz_project/bisholea/capstone/60M_human_gene_metadata.pkl")
print("Gene metadata shape:", gene_metadata.shape, flush=True)

# Prepare chunk file lists
count_files = sorted(glob("/mnt/projects/debruinz_project/bisholea/capstone/60M_human_counts/human_counts_*.npz"))
meta_files  = sorted(glob("/mnt/projects/debruinz_project/bisholea/capstone/60M_human_metadata/human_metadata_*.pkl"))
assert len(count_files) == len(meta_files), "Mismatched count and metadata files"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Instead of relying on X.shape[1], use known value from gene_metadata
model = Autoencoder(input_dim=gene_metadata.shape[0], latent_dim=256)
model.to(device)

# Initialize weights
init_positive_orthogonal_weights(
    model.encoder.weight,
    max_iter   = 5000,
    tol        = 5.0,
    patience   = 25,
    eps_change = 1e-4,
    verbose    = True
)

# Check orthogonality after initialization
with torch.no_grad():
    W = model.encoder.weight
    I = torch.eye(W.size(0), device=W.device)
    ortho_init = torch.sum((W @ W.T - I).pow(2)).item()
print(f"‖W Wᵀ – I‖² right after init  = {ortho_init:.4f}", flush=True)

# Initialize training
optimizer = optim.AdamW(model.parameters(), lr=1e-4)
loss_fn   = nn.MSELoss()
l1_lambda = 0.0
n_epochs  = 10
batch_size = 512
train_losses = []

# Training loop
# Loop over epochs
for epoch in range(n_epochs):
    epoch_start = time.time()
    print(f"\n Epoch {epoch + 1} / {n_epochs}", flush=True)
    model.train()
    total_loss = 0.0
    total_batches = 0

    # Loop over chunks
    for chunk_idx, (count_path, meta_path) in enumerate(zip(count_files, meta_files)):
        chunk_start = time.time()
        print(f"\n>>> Training on chunk {chunk_idx + 1} / {len(count_files)}", flush=True)

        # Load data
        t0 = time.time()
        X_sparse = sp.load_npz(count_path)
        t1 = time.time()

        metadata = pd.read_pickle(meta_path)
        t2 = time.time()

        # Include only normal disease samples
        mask = metadata["disease"] == "normal"
        X_sparse = X_sparse[mask.values]
        metadata = metadata[mask].reset_index(drop=True)
        t3 = time.time()

        assert X_sparse.shape[0] == metadata.shape[0], f"Mismatch in chunk {chunk_idx+1}: {X_sparse.shape[0]} vs {metadata.shape[0]},"
        assert X_sparse.shape[1] == gene_metadata.shape[0], f"Mismatch in chunk {chunk_idx+1}: {X_sparse.shape[1]} vs {gene_metadata.shape[0]}"
        
        dataset = SparseDataset(X_sparse)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        # Loop over batches
        for batch_indices in loader:
            batch_start = time.time()

            t_dense_start = time.time()
            batch_sparse = X_sparse[batch_indices.numpy()]                 # shape: (512, genes)
            x_batch = torch.tensor(batch_sparse.toarray(), dtype=torch.float32, device=device) # Convert to dense tensor
            t_dense_end = time.time()

            # if total_batches % 100 == 0:
            #     print(f"Sparse to dense time: {t_dense_end - t_dense_start:.3f}s", flush=True)
            
            optimizer.zero_grad()
            x_hat, _ = model(x_batch)
            recon_loss = loss_fn(x_hat, x_batch)
            l1_penalty = l1_lambda * model.encoder.weight.abs().sum()
            loss = recon_loss + l1_penalty

            loss.backward()
            optimizer.step()

            with torch.no_grad():
                model.encoder.weight.clamp_(min=0.0)

            batch_end = time.time()

            total_loss += loss.item()
            total_batches += 1

        chunk_end = time.time()
        # print(f"\n [Chunk {chunk_idx+1}] Times → .npz: {t1-t0:.2f}s | .pkl: {t2-t1:.2f}s | filter: {t3-t2:.2f}s | total chunk: {chunk_end - chunk_start:.2f}s", flush=True)

    avg_loss = total_loss / total_batches
    train_losses.append(avg_loss)

    with torch.no_grad():
        W = model.encoder.weight
        ortho = torch.sum((W @ W.T - torch.eye(W.size(0), device=W.device)).pow(2)).item()

    # Print epoch summary
    print(f"\n [Epoch {epoch+1}] AvgLoss = {avg_loss:.4f} | ‖W Wᵀ – I‖² = {ortho:.2f}", flush=True)

    # Save weights every 5 epochs or at the last epoch
    if (epoch + 1) % 5 == 0 or epoch == n_epochs - 1:
        W = model.encoder.weight.detach().cpu().numpy()
        np.save(f"/mnt/projects/debruinz_project/bisholea/60M_weights_epoch{epoch+1}.npy", W)
        print(f"Saved W matrix for epoch {epoch+1}")

    epoch_end = time.time()
    print(f"[Epoch {epoch+1}] Total time: {epoch_end - epoch_start:.2f}s", flush=True)

torch.save(model.state_dict(), '60M_model_state_dict.pth')

with open('/mnt/projects/debruinz_project/bisholea/60M_train_losses.pkl', 'wb') as f:
    pickle.dump(train_losses, f)

print("Training complete. Model and losses saved.")