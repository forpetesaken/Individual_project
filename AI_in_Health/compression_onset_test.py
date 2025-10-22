# compression_onset_test_unified.py (strict thresholds)
# Evaluate a saved model bundle on TEST CSV using thresholds saved in the bundle.
# Supports:
#  (A) PyTorch TinyCNN bundle: {"scaler","model_state_dict","win_var","delta_short","delta_pct","T_comp","T_reg",...}
#  (B) sklearn bundle dict:   {"scaler",("model"|"clf"),"win_var","delta_short","delta_pct","T_comp","T_reg",...}
#  (C) sklearn estimator/pipeline saved directly (must support predict_proba) + requires T_comp/T_reg in a sidecar dict is NOT supported here.

import argparse, joblib, numpy as np, pandas as pd
from sklearn.metrics import confusion_matrix, classification_report, average_precision_score, precision_recall_curve
import os
from pathlib import Path

try:
    import torch, torch.nn as nn
except Exception:
    torch = None
    nn = None

def make_features(glucose, win_var=15, d_short=5, d_pct=10):
    n = len(glucose); feats = np.zeros((n,6), float)
    for i in range(n):
        g = glucose[i]
        d1 = g - glucose[i-1] if i>0 else 0.0
        d5 = g - glucose[i-d_short] if i>=d_short else 0.0
        accel = d1 - (glucose[i-1]-glucose[i-2]) if i>1 else 0.0
        start = max(0, i-win_var)
        var15 = np.var(glucose[start:i+1]) if i>0 else 0.0
        pct10 = ((glucose[i-d_pct]-g)/glucose[i-d_pct]) if i>=d_pct and glucose[i-d_pct]!=0 else 0.0
        feats[i] = [g, d1, d5, accel, var15, pct10]
    return feats

class TinyCNN(nn.Module):
    def __init__(self, n_feat=6, n_classes=3):
        super().__init__()
        self.conv1 = nn.Conv1d(n_feat, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=3, padding=1)
        self.gap   = nn.AdaptiveAvgPool1d(1)
        self.fc    = nn.Linear(64, n_classes)
    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = self.gap(x).squeeze(-1)
        return self.fc(x)

def load_bundle(path):
    b = joblib.load(path)
    if isinstance(b, dict) and "model_state_dict" in b:
        return "torch", b
    if isinstance(b, dict) and ("model" in b or "clf" in b):
        return "sklearn", b
    # We require thresholds inside the bundle; raw sklearn objects without a dict are not supported here.
    raise TypeError("Unsupported bundle format. Expect a dict with PyTorch keys ('model_state_dict') "
                    "or sklearn keys ('model'|'clf').")

def main(args):
    # Create output directory if specified
    if args.out:
        output_dir = Path(args.out)
        output_dir.mkdir(parents=True, exist_ok=True)
        print(f"Output directory: {output_dir.absolute()}")
    else:
        output_dir = Path(".")
    
    # --- Load TEST CSV
    df = pd.read_csv(args.test_csv)
    df["glucose"] = df["glucose"].astype(str).str.upper().replace({"HIGH":401,"LOW":39}).astype(float)
    df = df.sort_values("timestamp").reset_index(drop=True)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    y_true = df["label"].astype(int).to_numpy()
    ts = df["timestamp"]

    # --- Load bundle
    mode, b = load_bundle(args.bundle)

    # --- Pull feature params (prefer args overrides if provided)
    def getp(key, default):
        return (getattr(args, key) if getattr(args, key) is not None
                else (b.get(key, default) if isinstance(b, dict) else default))
    win_var  = getp("win_var", 15)
    d_short  = getp("delta_short", 5)
    d_pct    = getp("delta_pct", 10)

    # --- Build features
    X = make_features(df["glucose"].to_numpy(float), win_var, d_short, d_pct)

    # --- Forward pass to get probabilities
    if mode == "torch":
        if torch is None:
            raise RuntimeError("PyTorch not available in this environment.")
        scaler = b["scaler"]
        Xs = scaler.transform(X)
        X_t = torch.tensor(Xs, dtype=torch.float32).unsqueeze(-1)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = TinyCNN(n_feat=X.shape[1], n_classes=3).to(device)
        model.load_state_dict(b["model_state_dict"])
        model.eval()
        with torch.no_grad():
            logits = model(X_t.to(device)).cpu()
            P = torch.softmax(logits, dim=1).numpy()

    elif mode == "sklearn":
        scaler = b.get("scaler", None)
        clf = b.get("model", b.get("clf", None))
        if scaler is not None:
            Xs = scaler.transform(X)
        else:
            Xs = X
        if not hasattr(clf, "predict_proba"):
            raise AttributeError("Loaded sklearn model has no predict_proba; enable probability=True or use a prob-compat model.")
        P_raw = clf.predict_proba(Xs)
        # Align columns to [0,1,2] using classes_
        classes = getattr(clf, "classes_", None)
        if classes is None:
            raise RuntimeError("Estimator missing classes_. Cannot align predict_proba columns.")
        if P_raw.shape[1] != len(classes):
            raise ValueError(f"predict_proba columns ({P_raw.shape[1]}) != len(classes_) ({len(classes)})")
        P = np.zeros((P_raw.shape[0], 3), dtype=float)
        for k, cls in enumerate(classes):
            if cls in (0,1,2):
                P[:, int(cls)] = P_raw[:, k]
            else:
                raise ValueError(f"Unexpected class label {cls}; expected only 0,1,2.")

    else:
        raise RuntimeError(f"Unknown mode {mode}")

    if P.shape[1] != 3:
        raise ValueError(f"Expected 3-class probabilities, got shape {P.shape}")

    p_none, p_comp, p_reg = P[:,0], P[:,1], P[:,2]

    # --- STRICT: Use thresholds only from bundle
    T_comp = b.get("T_comp", None)
    T_reg  = b.get("T_reg",  None)
    if T_comp is None or T_reg is None:
        raise ValueError("Bundle is missing thresholds T_comp/T_reg. Re-run training to save them into the bundle.")
    print(f"Using thresholds from bundle: T_comp={T_comp:.3f} | T_reg={T_reg:.3f}")

    # --- Thresholded predictions (prefer compression if both exceed)
    preds = np.zeros_like(y_true)
    for i in range(len(y_true)):
        if p_comp[i] > T_comp:
            preds[i] = 1
        elif p_reg[i] > T_reg:
            preds[i] = 2
        else:
            preds[i] = 0

    # --- Metrics
    ap_comp = average_precision_score((y_true==1).astype(int), p_comp)
    ap_reg  = average_precision_score((y_true==2).astype(int), p_reg)
    print(f"Average Precision - compression: {ap_comp:.3f}")
    print(f"Average Precision - regular    : {ap_reg:.3f}")

    cm = confusion_matrix(y_true, preds, labels=[0,1,2])
    print("\nConfusion matrix [rows=true, cols=pred]:\n", cm)
    print("\nClassification report (thresholded preds):\n",
          classification_report(y_true, preds, labels=[0,1,2], digits=3))

    # Save confusion matrix as figure
    if args.save_crosstab or args.out:
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(8, 6))
        im = ax.imshow(cm, interpolation='nearest', cmap='Blues')
        ax.figure.colorbar(im, ax=ax)
        
        # Add text annotations
        thresh = cm.max() / 2.
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, format(cm[i, j], 'd'),
                       ha="center", va="center",
                       color="white" if cm[i, j] > thresh else "black",
                       fontsize=12, fontweight='bold')
        
        ax.set_xlabel('Predicted Label', fontsize=12)
        ax.set_ylabel('True Label', fontsize=12)
        ax.set_title('Confusion Matrix (Crosstab)', fontsize=14, fontweight='bold')
        ax.set_xticks([0, 1, 2])
        ax.set_yticks([0, 1, 2])
        ax.set_xticklabels(['None', 'Compression', 'Regular'])
        ax.set_yticklabels(['None', 'Compression', 'Regular'])
        
        # Add accuracy text
        accuracy = np.trace(cm) / np.sum(cm)
        ax.text(0.5, -0.15, f'Accuracy: {accuracy:.3f}', 
                transform=ax.transAxes, ha='center', fontsize=11, fontweight='bold')
        
        crosstab_path = output_dir / "confusion_matrix.png"
        plt.savefig(crosstab_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved confusion matrix to {crosstab_path}")

    # Optional outputs
    if args.save_curves or args.out:
        import matplotlib.pyplot as plt
        for cls_name, y_bin, p_bin, fname in [
            ("compression", (y_true==1).astype(int), p_comp, "pr_comp.png"),
            ("regular",     (y_true==2).astype(int), p_reg,  "pr_reg.png"),
        ]:
            from sklearn.metrics import precision_recall_curve
            prec, rec, _ = precision_recall_curve(y_bin, p_bin)
            plt.figure()
            plt.plot(rec, prec)
            plt.xlabel("Recall"); plt.ylabel("Precision"); plt.title(f"PR Curve - {cls_name}")
            curve_path = output_dir / fname
            plt.savefig(curve_path, dpi=200, bbox_inches="tight")
            plt.close()
            print(f"Saved {curve_path}")

    if args.save_probs:
        probs_path = output_dir / args.save_probs if args.out else args.save_probs
        pd.DataFrame({
            "timestamp": ts, "y_true": y_true,
            "p_none": p_none, "p_comp": p_comp, "p_reg": p_reg,
            "pred": preds
        }).to_csv(probs_path, index=False)
        print(f"Saved probabilities to {probs_path}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--bundle", required=True, help="Path to the BEST bundle saved by training")
    ap.add_argument("--test_csv", required=True, help="Path to TEST CSV")
    ap.add_argument("--out", default="", help="Output directory for saved figures and files")
    ap.add_argument("--save_curves", action="store_true", help="Save PR curves")
    ap.add_argument("--save_crosstab", action="store_true", help="Save confusion matrix figure")
    ap.add_argument("--save_probs", default="", help="Filename to save probabilities CSV")
    # Optional overrides (use only if your bundle didn't store the exact values)
    ap.add_argument("--win_var", type=int, default=None)
    ap.add_argument("--delta_short", type=int, default=None)
    ap.add_argument("--delta_pct", type=int, default=None)
    args = ap.parse_args()
    main(args)
