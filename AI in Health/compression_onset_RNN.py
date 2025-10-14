"""
compression_onset.py
--------------------
End-to-end training & thresholding for a 3-class onset detector:
0 = none, 1 = compression-onset window, 2 = regular-onset window
"""

import argparse, joblib, numpy as np, pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupShuffleSplit, train_test_split, GroupKFold
from sklearn.metrics import (confusion_matrix, classification_report,
                             average_precision_score, precision_recall_curve)
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

# -----------------------
# Feature engineering
# -----------------------
def make_features(glucose: np.ndarray,
                  win_var: int = 15,
                  d_short: int = 5,
                  d_pct: int = 10) -> np.ndarray:
    """
    Create glucose-only features for each time index.

    Parameters are in samples (assume 1-sample = 1 minute).
    """
    n = len(glucose)
    feats = np.zeros((n, 6), dtype=float)
    for i in range(n):
        g_t = glucose[i]
        d1 = g_t - glucose[i-1] if i > 0 else 0.0
        d5 = g_t - glucose[i-d_short] if i >= d_short else 0.0
        accel = d1 - (glucose[i-1]-glucose[i-2]) if i > 1 else 0.0
        start = max(0, i-win_var)
        var15 = np.var(glucose[start:i+1]) if i > 0 else 0.0
        pct10 = ((glucose[i-d_pct] - g_t) / glucose[i-d_pct]) if i >= d_pct and glucose[i-d_pct]!=0 else 0.0
        feats[i] = [g_t, d1, d5, accel, var15, pct10]
    return feats

# -----------------------
# Threshold picker
# -----------------------
def pick_threshold(y_true_bin, p_class, timestamps, max_false_per_day=0.5):
    """
    Scan thresholds 0..1 to find the highest sensitivity s.t. FA/day <= target.
    y_true_bin: 1 for this class, 0 otherwise
    p_class: probabilities for this class
    timestamps: pd.Series of datetimes (sorted)
    """
    days = (timestamps.iloc[-1].normalize() - timestamps.iloc[0].normalize()).days + 1
    best = None
    for T in np.linspace(0, 1, 501):
        preds = p_class > T
        fa = ((preds) & (y_true_bin == 0)).sum()
        fa_per_day = fa / days
        sens = ((preds) & (y_true_bin == 1)).sum() / (y_true_bin.sum() + 1e-9)
        if fa_per_day <= max_false_per_day:
            if best is None or sens > best["sens"]:
                best = {"T": T, "sens": sens, "fa_per_day": fa_per_day}
    # Fallback: if nothing meets the FA constraint, return the threshold that minimizes FA/day
    if best is None:
        for T in np.linspace(0,1,501):
            preds = p_class > T
            fa = ((preds) & (y_true_bin == 0)).sum()
            fa_per_day = fa / days
            sens = ((preds) & (y_true_bin == 1)).sum() / (y_true_bin.sum() + 1e-9)
            if best is None or fa_per_day < best["fa_per_day"]:
                best = {"T": T, "sens": sens, "fa_per_day": fa_per_day}
    return best

# -----------------------
# Streaming helper
# -----------------------
class StreamingDetector:
    """
    Minimal runtime detector: keep last N glucose points, compute one feature row,
    get probs, apply thresholds.
    """
    def __init__(self, scaler, model, T_comp, T_reg,
                 win_var=15, d_short=5, d_pct=10,
                 refractory_minutes_comp=30, refractory_minutes_reg=15,
                 sample_period_min=1):
        self.scaler = scaler
        self.model = model
        self.T_comp = T_comp
        self.T_reg = T_reg
        self.buffer = []
        self.win_var = win_var
        self.d_short = d_short
        self.d_pct = d_pct
        self.ref_comp = refractory_minutes_comp
        self.ref_reg = refractory_minutes_reg
        self.sample_period = sample_period_min
        self.cooldown_comp = 0
        self.cooldown_reg = 0
        self.pending = False  # for compression confirmation

    def _features_last_point(self):
        g = np.array(self.buffer, dtype=float)
        f = make_features(g, self.win_var, self.d_short, self.d_pct)[-1].reshape(1,-1)
        return self.scaler.transform(f)

    def step(self, g_new):
        # cooldown bookkeeping
        self.cooldown_comp = max(0, self.cooldown_comp - self.sample_period)
        self.cooldown_reg  = max(0, self.cooldown_reg  - self.sample_period)

        # push new reading, ensure 120‑min buffer is full
        self.buffer.append(g_new)
        if len(self.buffer) < 24:  # 24 × 5‑min = 120 min
            return 0, None

        # get probabilities from CNN
        X_last = self._features_last_point()  # shape [1,n_feat]
        with torch.no_grad():
            logits = self.model(torch.tensor(X_last, dtype=torch.float32).unsqueeze(-1))
            p = torch.softmax(logits, dim=1).cpu().numpy()[0]  # [p0,p1,p2]

        # confirmation path (compression only)
        if self.pending:
            if p[1] >= self.T_comp:
                self.pending = False
                self.cooldown_comp = self.ref_comp
                return 1, p  # fire compression alert
            else:
                self.pending = False
                return 0, p

        # first‑pass detection
        if p[1] >= self.T_comp and self.cooldown_comp == 0:
            self.pending = True
            return 0, p

        if p[2] >= self.T_reg and self.cooldown_reg == 0:
            self.cooldown_reg = self.ref_reg
            return 2, p

        return 0, p

class TinyCNN(nn.Module):
    def __init__(self, n_feat=6, n_classes=3):
        super().__init__()
        self.conv1 = nn.Conv1d(n_feat, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=3, padding=1)
        self.gap   = nn.AdaptiveAvgPool1d(1)          # global avg‑pool
        self.fc    = nn.Linear(64, n_classes)

    def forward(self, x):             # x: [B, n_feat, T]  (T=1 here)
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = self.gap(x).squeeze(-1)   # [B, 64]
        return self.fc(x)             # logits



class TinyRNN(nn.Module):
    def __init__(self, n_feat=6, n_classes=3, hidden_size=32, num_layers=1):
        super().__init__()
        self.rnn = nn.GRU(input_size=n_feat, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, n_classes)

    def forward(self, x):
        # x: [B, n_feat, T] -> [B, T, n_feat]
        x = x.transpose(1, 2)
        _, h_n = self.rnn(x)
        out = h_n[-1]  # [B, hidden_size]
        return self.fc(out)  # logits



# -----------------------
# Main script
# -----------------------
def main(args):
    df = pd.read_csv(args.csv)

    # handle High / Low strings (case-insensitive)
    df["glucose"] = (
        df["glucose"]
          .astype(str)
          .str.upper()
          .replace({"HIGH": 401, "LOW": 39})
          .astype(float)
    )

    # Ensure sorting
    df = df.sort_values("timestamp").reset_index(drop=True)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    if "day_id" not in df.columns:
        df["day_id"] = df["timestamp"].dt.date

    glucose = df["glucose"].to_numpy(dtype=float)
    y = df["label"].astype(int).to_numpy()

    # Build features
    X = make_features(glucose,
                      win_var=args.win_var,
                      d_short=args.delta_short,
                      d_pct=args.delta_pct)

    # --- GroupKFold cross-validation ---
    groups = df["day_id"] if "day_id" in df.columns else None
    if groups is not None:
        ap_comp_list, ap_reg_list = [], []
        all_loss = []  # For combined loss plot
        gkf = GroupKFold(n_splits=5)
        for fold, (tr, va) in enumerate(gkf.split(X, y, groups=groups)):
            print(f"\n=== Fold {fold+1} ===")
            X_train, X_val = X[tr], X[va]
            y_train, y_val = y[tr], y[va]
            ts_val = df["timestamp"].iloc[va]

            # Scale
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_val   = scaler.transform(X_val)

            # --- CNN: reshape and convert to torch tensors ---
            # CNN expects [B, n_feat, T], here T=1
            X_train_torch = torch.tensor(X_train, dtype=torch.float32).unsqueeze(-1)  # [N, n_feat, 1]
            y_train_torch = torch.tensor(y_train, dtype=torch.long)
            X_val_torch   = torch.tensor(X_val, dtype=torch.float32).unsqueeze(-1)
            y_val_torch   = torch.tensor(y_val, dtype=torch.long)

            train_ds = TensorDataset(X_train_torch, y_train_torch)
            val_ds   = TensorDataset(X_val_torch, y_val_torch)
            train_loader = DataLoader(train_ds, batch_size=128, shuffle=True)
            val_loader   = DataLoader(val_ds, batch_size=256)

            # --- Model ---
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model = TinyRNN(n_feat=X_train.shape[1], n_classes=3).to(device)
            optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
            # Set class weights: prefer class 2 (regular/true low) > class 1 (compression) > class 0 (none)
            weight_none = 1.0
            weight_comp = 5.0    # compression
            weight_reg  = 10.0   # regular/true low (highest penalty for missing)
            class_weights = torch.tensor([weight_none, weight_comp, weight_reg]).to(device)
            criterion = nn.CrossEntropyLoss(weight=class_weights)

            # --- Training loop ---
            loss_hist = []
            for epoch in range(args.epochs):
                running = 0.0
                for xb, yb in train_loader:
                    xb, yb = xb.to(device), yb.to(device)
                    optimizer.zero_grad()
                    loss = criterion(model(xb), yb)
                    loss.backward()
                    optimizer.step()
                    running += loss.item() * xb.size(0)
                epoch_loss = running / len(train_ds)
                loss_hist.append(epoch_loss)
            all_loss.append(loss_hist)

            # --- Validation: get probabilities ---
            model.eval()
            with torch.no_grad():
                logits_val = []
                for xb, _ in val_loader:
                    xb = xb.to(device)
                    logits_val.append(model(xb).cpu())
                logits_val = torch.cat(logits_val, dim=0)
                P_val = torch.softmax(logits_val, dim=1).numpy()
            p_comp = P_val[:,1]
            p_reg  = P_val[:,2]

            # AP scores (optional but nice on poster)
            from sklearn.metrics import average_precision_score
            ap_comp = average_precision_score((y_val==1).astype(int), p_comp)
            ap_reg  = average_precision_score((y_val==2).astype(int), p_reg)
            print(f"Average Precision - compression: {ap_comp:.3f}")
            print(f"Average Precision - regular    : {ap_reg:.3f}")

            # Thresholds
            best_comp = pick_threshold((y_val==1).astype(int), p_comp, ts_val,
                                       max_false_per_day=args.fa_day_comp)
            best_reg  = pick_threshold((y_val==2).astype(int), p_reg , ts_val,
                                       max_false_per_day=args.fa_day_reg)

            print("Chosen thresholds:")
            print(f"  Compression -> T={best_comp['T']:.3f}, Sens={best_comp['sens']:.3f}, FA/day={best_comp['fa_per_day']:.3f}")
            print(f"  Regular     -> T={best_reg['T']:.3f}, Sens={best_reg['sens']:.3f}, FA/day={best_reg['fa_per_day']:.3f}")

            # Confusion matrix at those thresholds
            preds_val = np.zeros_like(y_val)
            # prefer compression if both exceed
            for i in range(len(y_val)):
                if p_comp[i] > best_comp["T"]:
                    preds_val[i] = 1
                elif p_reg[i] > best_reg["T"]:
                    preds_val[i] = 2
                else:
                    preds_val[i] = 0

            cm = confusion_matrix(y_val, preds_val, labels=[0,1,2])
            print("\nConfusion matrix [rows=true, cols=pred]:\n", cm)
            print("\nClassification report (using thresholded preds):\n",
                  classification_report(y_val, preds_val, labels=[0,1,2], digits=3))
            # Optionally: aggregate or save results per fold
            ap_comp_list.append(ap_comp)
            ap_reg_list.append(ap_reg)

            # --- Plot loss curve for this fold ---
            import matplotlib.pyplot as plt
            plt.figure()
            plt.plot(range(1, len(loss_hist)+1), loss_hist, marker='o')
            plt.xlabel("Epoch")
            plt.ylabel("Training loss")
            plt.title(f"Fold {fold+1} {args.suffix} – loss curve")
            plt.grid(True)
            plt.savefig(f"loss_fold{fold+1}_{args.suffix}.png", dpi=200, bbox_inches="tight")
            print(f"Saved loss_fold{fold+1}_{args.suffix}.png")

        # Combined loss plot for all folds
        plt.figure()
        for i, lh in enumerate(all_loss):
            plt.plot(range(1, len(lh)+1), lh, label=f"Fold {i+1}")
        plt.xlabel("Epoch")
        plt.ylabel("Training loss")
        plt.legend()
        plt.title("Epoch vs Loss – all folds")
        plt.grid(True)
        plt.savefig(f"loss_all_folds_{args.suffix}.png", dpi=200, bbox_inches="tight")
        print(f"Saved loss_all_folds_{args.suffix}.png")

        # Only print averages if at least one fold ran
        if ap_comp_list and ap_reg_list:
            print("\n=== 5‑fold averages ===")
            print(f"AP‑comp   : {np.mean(ap_comp_list):.3f} ± {np.std(ap_comp_list):.3f}")
            print(f"AP‑regular: {np.mean(ap_reg_list):.3f} ± {np.std(ap_reg_list):.3f}")
        else:
            print("\n=== 5‑fold averages ===")
            print("No folds ran: check your data and group assignments.")
    else:
        # Train/val split
        if "group_id" in df.columns:
            groups = df["group_id"].astype(str).to_numpy()
            gss = GroupShuffleSplit(test_size=0.2, n_splits=1, random_state=42)
            tr, va = next(gss.split(X, y, groups))
        else:
            tr, va = train_test_split(np.arange(len(y)), test_size=0.2,
                                      stratify=y, random_state=42)
            groups = None

        X_train, X_val = X[tr], X[va]
        y_train, y_val = y[tr], y[va]
        ts_val = df["timestamp"].iloc[va]

        # Scale
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_val   = scaler.transform(X_val)

        # --- CNN: reshape and convert to torch tensors ---
        # CNN expects [B, n_feat, T], here T=1
        X_train_torch = torch.tensor(X_train, dtype=torch.float32).unsqueeze(-1)  # [N, n_feat, 1]
        y_train_torch = torch.tensor(y_train, dtype=torch.long)
        X_val_torch   = torch.tensor(X_val, dtype=torch.float32).unsqueeze(-1)
        y_val_torch   = torch.tensor(y_val, dtype=torch.long)

        train_ds = TensorDataset(X_train_torch, y_train_torch)
        val_ds   = TensorDataset(X_val_torch, y_val_torch)
        train_loader = DataLoader(train_ds, batch_size=128, shuffle=True)
        val_loader   = DataLoader(val_ds, batch_size=256)

        # --- Model ---
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = TinyCNN(n_feat=X_train.shape[1], n_classes=3).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        # Set class weights: prefer class 2 (regular/true low) > class 1 (compression) > class 0 (none)
        weight_none = 1.0
        weight_comp = 5.0    # compression
        weight_reg  = 10.0   # regular/true low (highest penalty for missing)
        class_weights = torch.tensor([weight_none, weight_comp, weight_reg]).to(device)
        criterion = nn.CrossEntropyLoss(weight=class_weights)

        # --- Training loop ---
        n_epochs = 20
        model.train()
        for epoch in range(n_epochs):
            total_loss = 0
            for xb, yb in train_loader:
                xb, yb = xb.to(device), yb.to(device)
                optimizer.zero_grad()
                logits = model(xb)
                loss = criterion(logits, yb)
                loss.backward()
                optimizer.step()
                total_loss += loss.item() * xb.size(0)
            # Optionally print loss
            # print(f"Epoch {epoch+1}/{n_epochs}, Loss: {total_loss/len(train_ds):.4f}")

        # --- Validation: get probabilities ---
        model.eval()
        with torch.no_grad():
            logits_val = []
            for xb, _ in val_loader:
                xb = xb.to(device)
                logits_val.append(model(xb).cpu())
            logits_val = torch.cat(logits_val, dim=0)
            P_val = torch.softmax(logits_val, dim=1).numpy()
        p_comp = P_val[:,1]
        p_reg  = P_val[:,2]

        # AP scores (optional but nice on poster)
        from sklearn.metrics import average_precision_score
        ap_comp = average_precision_score((y_val==1).astype(int), p_comp)
        ap_reg  = average_precision_score((y_val==2).astype(int), p_reg)
        print(f"Average Precision - compression: {ap_comp:.3f}")
        print(f"Average Precision - regular    : {ap_reg:.3f}")

        # Thresholds
        best_comp = pick_threshold((y_val==1).astype(int), p_comp, ts_val,
                                   max_false_per_day=args.fa_day_comp)
        best_reg  = pick_threshold((y_val==2).astype(int), p_reg , ts_val,
                                   max_false_per_day=args.fa_day_reg)

        print("Chosen thresholds:")
        print(f"  Compression -> T={best_comp['T']:.3f}, Sens={best_comp['sens']:.3f}, FA/day={best_comp['fa_per_day']:.3f}")
        print(f"  Regular     -> T={best_reg['T']:.3f}, Sens={best_reg['sens']:.3f}, FA/day={best_reg['fa_per_day']:.3f}")

        # Confusion matrix at those thresholds
        preds_val = np.zeros_like(y_val)
        # prefer compression if both exceed
        for i in range(len(y_val)):
            if p_comp[i] > best_comp["T"]:
                preds_val[i] = 1
            elif p_reg[i] > best_reg["T"]:
                preds_val[i] = 2
            else:
                preds_val[i] = 0

        cm = confusion_matrix(y_val, preds_val, labels=[0,1,2])
        print("\nConfusion matrix [rows=true, cols=pred]:\n", cm)
        print("\nClassification report (using thresholded preds):\n",
              classification_report(y_val, preds_val, labels=[0,1,2], digits=3))

        # Save artifacts
        bundle = {
            "scaler": scaler,
            "model_state_dict": model.state_dict(),
            "T_comp": best_comp["T"],
            "T_reg":  best_reg["T"],
            "win_var": args.win_var,
            "delta_short": args.delta_short,
            "delta_pct": args.delta_pct
        }
        joblib.dump(bundle, args.out)
        print(f"\nSaved model bundle to {args.out}")

        # Optional: save PR curves for poster
        if args.save_curves:
            import matplotlib.pyplot as plt
            for cls_name, y_bin, p_bin, fname in [
                    ("compression", (y_val==1).astype(int), p_comp, "pr_comp.png"),
                    ("regular",     (y_val==2).astype(int), p_reg,  "pr_reg.png")]:
                prec, rec, thr = precision_recall_curve(y_bin, p_bin)
                plt.figure()
                plt.plot(rec, prec)
                plt.xlabel("Recall")
                plt.ylabel("Precision")
                plt.title(f"PR Curve - {cls_name}")
                plt.savefig(fname, dpi=200, bbox_inches="tight")
                print(f"Saved {fname}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True, help="Path to labeled CSV")
    parser.add_argument("--suffix", required=True, help="Suffix for output files")
    parser.add_argument("--out", default="model_bundle.joblib", help="Output model file")
    parser.add_argument("--fa_day_comp", type=float, default=0.5, help="Max false alerts/day for compression")
    parser.add_argument("--fa_day_reg",  type=float, default=1.0, help="Max false alerts/day for regular")
    parser.add_argument("--win_var", type=int, default=15, help="Window (samples) for rolling variance")
    parser.add_argument("--delta_short", type=int, default=5,  help="Samples for short delta (d5)")
    parser.add_argument("--delta_pct", type=int, default=10, help="Samples for %%drop window")
    parser.add_argument("--save_curves", action="store_true", help="Save PR curve PNGs")
    parser.add_argument("--epochs", type=int, default=30, help="Number of training epochs")
    args = parser.parse_args()
    main(args)

