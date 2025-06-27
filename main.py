#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 27 Jun 2025 -- to be updated...
@author: Babak Ahmadi
"""

# ------------------------------------------------------------------
# Imports
# ------------------------------------------------------------------
import os, argparse, numpy as np, torch, torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import mean_absolute_error, r2_score
from densenet3d import DenseNet3D

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[INFO] device: {device}  |  GPUs: {torch.cuda.device_count()}")

# ------------------------------------------------------------------
# GLOBAL VARIABLES  
# ------------------------------------------------------------------
learning_rate = 5e-6      # initial LR
lru           = 0.7       # LR decay per outer iteration
batch_size    = 8
fit_iter      = 5         # outer iterations
fit_ep        = 15        # epochs per outer iteration
patience      = 6         # early-stopping patience
outdir        = "checkpoints"

# ------------------------------------------------------------------
# Helper: save checkpoint
# ------------------------------------------------------------------
def save_model(model, val_loss, path):
    torch.save({"model_state_dict": model.state_dict(),
                "val_loss": val_loss}, path)
    print(f"[INFO] saved {path} (val MAE = {val_loss:.4f})")

# ------------------------------------------------------------------
# Helper: make DataLoader from NumPy arrays
# ------------------------------------------------------------------
def make_loader(x, y, shuffle):
    xt = torch.tensor(x, dtype=torch.float32).permute(0, 4, 1, 2, 3)
    yt = torch.tensor(y, dtype=torch.float32).unsqueeze(1)
    return DataLoader(TensorDataset(xt, yt),
                      batch_size=batch_size, shuffle=shuffle,
                      drop_last=False)

# ------------------------------------------------------------------
# TRAINING
# ------------------------------------------------------------------
def train(x_train, y_train, x_val, y_val):
    os.makedirs(outdir, exist_ok=True)

    tr_loader = make_loader(x_train, y_train, True)
    va_loader = make_loader(x_val,   y_val,   False)

    model = DenseNet3D().to(device)
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)

    best_val = float("inf")
    lr = learning_rate

    for it in range(fit_iter):
        if it: lr *= lru
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        print(f"\n[Iter {it+1}/{fit_iter}] lr = {lr:.2e}")
        wait = 0

        for ep in range(fit_ep):
            # ---- train ----
            model.train(); run_loss = 0.0
            for X, y in tr_loader:
                X, y = X.to(device), y.to(device)
                optimizer.zero_grad()
                loss = nn.functional.l1_loss(model(X), y)
                loss.backward(); optimizer.step()
                run_loss += loss.item() * len(X)
            tr_loss = run_loss / len(tr_loader.dataset)

            # ---- val ----
            model.eval(); val_loss = 0.0
            with torch.no_grad():
                for X, y in va_loader:
                    X, y = X.to(device), y.to(device)
                    val_loss += nn.functional.l1_loss(model(X), y).item() * len(X)
            val_loss /= len(va_loader.dataset)
            print(f"  epoch {ep+1:02}/{fit_ep}  train {tr_loss:.4f}  val {val_loss:.4f}")

            if val_loss < best_val:
                best_val = val_loss; wait = 0
                save_model(model, best_val, f"{outdir}/best_model.pt")
            else:
                wait += 1
                if wait >= patience:
                    print("  early stop")
                    return

# ------------------------------------------------------------------
# PREDICTION
# ------------------------------------------------------------------
def predict(x_test, weights, y_true=None):
    model = DenseNet3D().to(device)
    ckpt = torch.load(weights, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    te_loader = make_loader(x_test, np.zeros(len(x_test)), False)
    preds = []
    with torch.no_grad():
        for X, _ in te_loader:
            preds.append(model(X.to(device)).cpu().numpy())
    preds = np.concatenate(preds)
    np.save("predictions.npy", preds)
    print("[INFO] saved predictions.npy")

    if y_true is not None:
        mae = mean_absolute_error(y_true, preds)
        r2  = r2_score(y_true, preds)
        print(f"MAE = {mae:.3f}  |  R² = {r2:.3f}")

# ------------------------------------------------------------------
# CLI
# ------------------------------------------------------------------
if __name__ == "__main__":
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="mode", required=True)

    t = sub.add_parser("train")
    t.add_argument("--x_train"); t.add_argument("--y_train")
    t.add_argument("--x_val");   t.add_argument("--y_val")

    e = sub.add_parser("predict")
    e.add_argument("--x_test"); e.add_argument("--weights")
    e.add_argument("--y_test", help="optional true ages")

    args = p.parse_args()

    if args.mode == "train":
        Xtr = np.load(args.x_train); Ytr = np.load(args.y_train)
        Xva = np.load(args.x_val);   Yva = np.load(args.y_val)
        train(Xtr, Ytr, Xva, Yva)

    else:  # predict
        Xte = np.load(args.x_test)
        y_true = np.load(args.y_test) if args.y_test else None
        predict(Xte, args.weights, y_true)
