"""
compression_onset.py
--------------------
End-to-end training & thresholding for a 3-class onset detector:
0 = none, 1 = compression-onset window, 2 = regular-onset window
"""

import argparse, numpy as np, pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupShuffleSplit, train_test_split, GroupKFold
from sklearn.metrics import (confusion_matrix, classification_report,
                             average_precision_score, precision_recall_curve)
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import average_precision_score
import optuna


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
    def __init__(self, n_feat=6, n_classes=3, conv1_out=32, conv2_out=64, kernel_size=3, dropout=0.0):
        super().__init__()
        self.conv1 = nn.Conv1d(n_feat, conv1_out, kernel_size=kernel_size, padding=kernel_size//2)
        self.conv2 = nn.Conv1d(conv1_out, conv2_out, kernel_size=kernel_size, padding=kernel_size//2)
        self.gap   = nn.AdaptiveAvgPool1d(1)
        self.fc    = nn.Linear(conv2_out, n_classes)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = self.gap(x).squeeze(-1)
        x = self.dropout(x)
        return self.fc(x)

def model_training(
    csv,
    fa_day_comp=0.5,
    fa_day_reg=1.0,
    win_var=15,
    delta_short=5,
    delta_pct=10,
    save_curves=False,
    epochs=30,
    suffix="",
    cnn_params=None
):
    df = pd.read_csv(csv)
    df["glucose"] = (
        df["glucose"]
            .astype(str)
            .str.upper()
            .replace({"HIGH": 401, "LOW": 39})
            .astype(float)
    )
    df = df.sort_values("timestamp").reset_index(drop=True)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    if "day_id" not in df.columns:
        df["day_id"] = df["timestamp"].dt.date

    glucose = df["glucose"].to_numpy(dtype=float)
    y = df["label"].astype(int).to_numpy()
    X = make_features(glucose, win_var=win_var, d_short=delta_short, d_pct=delta_pct)

    groups = df["day_id"] if "day_id" in df.columns else None
    ap_comp_list, ap_reg_list = [], []
    sens_comp_list, fa_comp_list = [], []
    sens_reg_list, fa_reg_list = [], []
    all_loss = []
    gkf = GroupKFold(n_splits=5)
    for fold, (tr, va) in enumerate(gkf.split(X, y, groups=groups)):
        X_train, X_val = X[tr], X[va]
        y_train, y_val = y[tr], y[va]
        ts_val = df["timestamp"].iloc[va]

        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_val   = scaler.transform(X_val)

        X_train_torch = torch.tensor(X_train, dtype=torch.float32).unsqueeze(-1)
        y_train_torch = torch.tensor(y_train, dtype=torch.long)
        X_val_torch   = torch.tensor(X_val, dtype=torch.float32).unsqueeze(-1)
        y_val_torch   = torch.tensor(y_val, dtype=torch.long)

        train_ds = TensorDataset(X_train_torch, y_train_torch)
        val_ds   = TensorDataset(X_val_torch, y_val_torch)
        train_loader = DataLoader(train_ds, batch_size=cnn_params.get("batch_size", 128), shuffle=True)
        val_loader   = DataLoader(val_ds, batch_size=256)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = TinyCNN(
            n_feat=X_train.shape[1],
            n_classes=3,
            conv1_out=cnn_params.get("conv1_out", 32),
            conv2_out=cnn_params.get("conv2_out", 64),
            kernel_size=cnn_params.get("kernel_size", 3),
            dropout=cnn_params.get("dropout", 0.0)
        ).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=cnn_params.get("lr", 1e-3))
        weight_none = 1.0
        weight_comp = 5.0
        weight_reg  = 10.0
        class_weights = torch.tensor([weight_none, weight_comp, weight_reg]).to(device)
        criterion = nn.CrossEntropyLoss(weight=class_weights)

        loss_hist = []
        for epoch in range(epochs):
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

        
        ap_comp = average_precision_score((y_val==1).astype(int), p_comp)
        ap_reg  = average_precision_score((y_val==2).astype(int), p_reg)

        best_comp = pick_threshold((y_val==1).astype(int), p_comp, ts_val, max_false_per_day=fa_day_comp)
        best_reg  = pick_threshold((y_val==2).astype(int), p_reg , ts_val, max_false_per_day=fa_day_reg)

        preds_val = np.zeros_like(y_val)
        for i in range(len(y_val)):
            if p_comp[i] > best_comp["T"]:
                preds_val[i] = 1
            elif p_reg[i] > best_reg["T"]:
                preds_val[i] = 2
            else:
                preds_val[i] = 0

        cm = confusion_matrix(y_val, preds_val, labels=[0,1,2])
        cr = classification_report(y_val, preds_val, labels=[0,1,2], digits=3)

        ap_comp_list.append(ap_comp)
        ap_reg_list.append(ap_reg)
        sens_comp_list.append(best_comp["sens"])
        fa_comp_list.append(best_comp["fa_per_day"])
        sens_reg_list.append(best_reg["sens"])
        fa_reg_list.append(best_reg["fa_per_day"])

    metrics = {
        "ap_comp": np.mean(ap_comp_list),
        "ap_reg": np.mean(ap_reg_list),
        "sens_comp": np.mean(sens_comp_list),
        "fa_per_day_comp": np.mean(fa_comp_list),
        "sens_reg": np.mean(sens_reg_list),
        "fa_per_day_reg": np.mean(fa_reg_list),
        "confusion_matrix": cm,
        "classification_report": cr,
        "best_comp_threshold": np.mean([b["T"] for b in [best_comp]]),
        "best_reg_threshold": np.mean([b["T"] for b in [best_reg]])
    }
    # Return the last trained model and scaler for simplicity
    return model, scaler, metrics

def objective(trial, csv, fa_day_comp, fa_day_reg, win_var, delta_short, delta_pct, save_curves, epochs, suffix):
    cnn_params = {
        "conv1_out": trial.suggest_int("conv1_out", 16, 64),
        "conv2_out": trial.suggest_int("conv2_out", 32, 128),
        "kernel_size": trial.suggest_int("kernel_size", 2, 5),
        "dropout": trial.suggest_float("dropout", 0.0, 0.5),
        "lr": trial.suggest_float("lr", 1e-4, 1e-2, log=True),
        "batch_size": trial.suggest_categorical("batch_size", [64, 128, 256])
    }
    _, _, metrics = model_training(
        csv=csv,
        fa_day_comp=fa_day_comp,
        fa_day_reg=fa_day_reg,
        win_var=win_var,
        delta_short=delta_short,
        delta_pct=delta_pct,
        save_curves=save_curves,
        epochs=epochs,
        suffix=suffix,
        cnn_params=cnn_params
    )
    # Optimize for average precision of compression onset
    return metrics["ap_comp"]

def run_hyperparameter_tuning(
    csv,
    fa_day_comp=0.5,
    fa_day_reg=1.0,
    win_var=15,
    delta_short=5,
    delta_pct=10,
    save_curves=False,
    epochs=30,
    suffix="",
    n_trials=20
):
    study = optuna.create_study(direction="maximize")
    study.optimize(lambda trial: objective(
        trial, csv, fa_day_comp, fa_day_reg, win_var, delta_short, delta_pct, save_curves, epochs, suffix
    ), n_trials=n_trials)
    best_params = study.best_params
    print("Best hyperparameters:", best_params)
    model, scaler, metrics = model_training(
        csv=csv,
        fa_day_comp=fa_day_comp,
        fa_day_reg=fa_day_reg,
        win_var=win_var,
        delta_short=delta_short,
        delta_pct=delta_pct,
        save_curves=save_curves,
        epochs=epochs,
        suffix=suffix,
        cnn_params=best_params
    )
    print("Best metrics:", metrics)

    return model, scaler, metrics, best_params

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True, help="Path to labeled CSV")
    parser.add_argument("--fa_day_comp", type=float, default=0.5, help="Max false alerts/day for compression")
    parser.add_argument("--fa_day_reg",  type=float, default=1.0, help="Max false alerts/day for regular")
    parser.add_argument("--win_var", type=int, default=15, help="Window (samples) for rolling variance")
    parser.add_argument("--delta_short", type=int, default=5,  help="Samples for short delta (d5)")
    parser.add_argument("--delta_pct", type=int, default=10, help="Samples for %%drop window")
    parser.add_argument("--save_curves", type=bool, default=False, help="Save PR curve PNGs")
    parser.add_argument("--epochs", type=int, default=30, help="Number of training epochs")
    parser.add_argument("--suffix", type=str, default="", help="Suffix for output file name")
    parser.add_argument("--n_trials", type=int, default=20, help="Number of Optuna trials")
    args = parser.parse_args()

    model, scaler, metrics, best_params = run_hyperparameter_tuning(
        csv=args.csv,
        fa_day_comp=args.fa_day_comp,
        fa_day_reg=args.fa_day_reg,
        win_var=args.win_var,
        delta_short=args.delta_short,
        delta_pct=args.delta_pct,
        save_curves=args.save_curves,
        epochs=args.epochs,
        suffix=args.suffix,
        n_trials=args.n_trials
    )

    # Retrain with best hyperparameters and save curves
    model, scaler, metrics = model_training(
        csv=args.csv,
        fa_day_comp=args.fa_day_comp,
        fa_day_reg=args.fa_day_reg,
        win_var=args.win_var,
        delta_short=args.delta_short,
        delta_pct=args.delta_pct,
        save_curves=True,
        epochs=args.epochs,
        suffix=args.suffix,
        cnn_params=best_params
    )
    print("Final metrics:", metrics)
    # Save best hyperparameters to file
    with open(f"{args.suffix}_best_params", "w") as f:
        f.write(str(best_params))
    torch.save(model.state_dict(), f"{args.suffix}_model")
