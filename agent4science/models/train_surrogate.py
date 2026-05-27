"""General-purpose surrogate model training script.

Trains a ResNetDNN surrogate model for any domain described by a DomainSpec.
Works with CSV datasets: columns 0..input_dim-1 = features, columns
input_dim.. = targets.
"""

from __future__ import annotations

import os
import argparse
import time

import numpy as np
import torch
import torch.nn as nn

from .resnet_model import ResNetDNN
from ..domain_spec import DomainSpec


def train_surrogate(
    domain: DomainSpec,
    train_data: np.ndarray,
    val_data: np.ndarray | None = None,
    epochs: int = 500,
    batch_size: int = 256,
    lr: float = 1e-3,
    device: torch.device | None = None,
    save_dir: str = "saved_models",
    patience: int = 50,
    verbose: bool = True,
) -> tuple[ResNetDNN, dict]:
    """Train a ResNetDNN surrogate model for a given domain.

    Args:
        domain: DomainSpec describing input/output dimensions.
        train_data: (N, input_dim + output_dim) numpy array.
        val_data: Optional validation data.
        epochs: Max training epochs.
        batch_size: Batch size.
        lr: Initial learning rate.
        device: Torch device.
        save_dir: Directory to save best model checkpoint.
        patience: Early stopping patience.
        verbose: Print progress.

    Returns:
        (trained_model, training_stats).
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    input_dim = domain.input_dim
    output_dim = domain.output_dim
    hidden_dim = domain.hidden_dim
    num_blocks = domain.num_blocks

    # Split features and targets
    X_train = torch.tensor(train_data[:, :input_dim], dtype=torch.float32)
    y_train = torch.tensor(train_data[:, input_dim:input_dim + output_dim],
                           dtype=torch.float32)

    if val_data is not None:
        X_val = torch.tensor(val_data[:, :input_dim], dtype=torch.float32)
        y_val = torch.tensor(val_data[:, input_dim:input_dim + output_dim],
                             dtype=torch.float32)
    else:
        # Use 20% of training as validation
        n_val = int(len(X_train) * 0.2)
        indices = torch.randperm(len(X_train))
        val_indices = indices[:n_val]
        train_indices = indices[n_val:]
        X_val = X_train[val_indices]
        y_val = y_train[val_indices]
        X_train = X_train[train_indices]
        y_train = y_train[train_indices]

    if verbose:
        print(f"Training {domain.name} surrogate: {input_dim}→{output_dim}")
        print(f"  Train: {len(X_train)}, Val: {len(X_val)}")
        print(f"  Hidden: {hidden_dim}, Blocks: {num_blocks}, Device: {device}")

    # Model
    model = ResNetDNN(
        input_dim=input_dim, output_dim=output_dim,
        hidden_dim=hidden_dim, num_blocks=num_blocks,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=20
    )
    criterion = nn.MSELoss()

    best_val_loss = float("inf")
    best_epoch = 0
    patience_counter = 0
    history = []

    t_start = time.time()

    for epoch in range(1, epochs + 1):
        model.train()
        # Shuffle
        perm = torch.randperm(len(X_train))
        X_shuffled = X_train[perm]
        y_shuffled = y_train[perm]

        total_loss = 0.0
        n_batches = 0

        for i in range(0, len(X_train), batch_size):
            xb = X_shuffled[i:i + batch_size].to(device)
            yb = y_shuffled[i:i + batch_size].to(device)

            optimizer.zero_grad()
            pred = model(xb)
            loss = criterion(pred, yb)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        train_loss = total_loss / max(n_batches, 1)

        # Validation
        model.eval()
        with torch.no_grad():
            val_pred = model(X_val.to(device))
            val_loss = criterion(val_pred, y_val.to(device)).item()

            # R² score
            ss_res = ((y_val.to(device) - val_pred) ** 2).sum().item()
            ss_tot = ((y_val.to(device) - y_val.mean()) ** 2).sum().item()
            val_r2 = 1.0 - ss_res / max(ss_tot, 1e-10)

        history.append({
            "epoch": epoch, "train_loss": train_loss,
            "val_loss": val_loss, "val_r2": val_r2,
        })

        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            patience_counter = 0

            # Save checkpoint
            os.makedirs(save_dir, exist_ok=True)
            checkpoint_path = os.path.join(
                save_dir, f"surrogate_{domain.name}.pth"
            )
            torch.save(model.state_dict(), checkpoint_path)
        else:
            patience_counter += 1

        if verbose and epoch % 50 == 0:
            print(f"  Epoch {epoch:4d} | Train Loss: {train_loss:.6f} | "
                  f"Val Loss: {val_loss:.6f} | Val R²: {val_r2:.4f}")

        if patience_counter >= patience:
            if verbose:
                print(f"  Early stopping at epoch {epoch}")
            break

    t_end = time.time()
    if verbose:
        print(f"  Done in {t_end - t_start:.1f}s")
        print(f"  Best epoch: {best_epoch}, Val R²: {history[best_epoch - 1]['val_r2']:.4f}")

    # Load best model
    checkpoint_path = os.path.join(save_dir, f"surrogate_{domain.name}.pth")
    if os.path.exists(checkpoint_path):
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))

    model.eval()
    for p in model.parameters():
        p.requires_grad = False

    stats = {
        "best_epoch": best_epoch,
        "best_val_loss": best_val_loss,
        "best_val_r2": history[best_epoch - 1]["val_r2"] if history else 0.0,
        "total_time_s": t_end - t_start,
        "history": history,
        "checkpoint_path": checkpoint_path,
    }

    return model, stats


# ── CLI ──

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train surrogate model for a domain.")
    parser.add_argument("--domain", type=str, required=True,
                        choices=["acoustic", "airfoil", "concrete"],
                        help="Domain name.")
    parser.add_argument("--data", type=str, required=True,
                        help="Path to CSV data file.")
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--save-dir", type=str, default="saved_models")

    args = parser.parse_args()

    domain = DomainSpec.from_yaml(
        os.path.join("agent4science", "configs", f"{args.domain}.yaml")
    )

    data = np.loadtxt(args.data, delimiter=",", skiprows=1)
    print(f"Loaded {len(data)} samples from {args.data}")

    model, stats = train_surrogate(
        domain, data,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        save_dir=args.save_dir,
    )

    print(f"\nFinal Val R²: {stats['best_val_r2']:.4f}")
    print(f"Model saved: {stats['checkpoint_path']}")
