"""
Model Comparison Script
======================

Compare CNN model against simple baseline approaches:
- Logistic Regression (statistical baseline)
- Rule-based model (clinical baseline)

Simplified version to avoid dependency issues.
"""

import argparse
import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupShuffleSplit, train_test_split, GroupKFold
from sklearn.metrics import (confusion_matrix, classification_report,
                             average_precision_score, precision_recall_curve)
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt
from pathlib import Path
import time
from collections import defaultdict

# Import original CNN and feature functions
from compression_onset import make_features, pick_threshold, TinyCNN, StreamingDetector
from baseline_models import get_all_baselines_wrapped


# Removed PyTorch baseline training since we only have sklearn baselines now


def train_cnn_model(X_train, y_train, X_val, y_val, epochs=30, device="cpu"):
    """Train the original CNN model."""
    
    # Reshape for CNN (add sequence dimension)
    X_train_torch = torch.tensor(X_train, dtype=torch.float32).unsqueeze(-1)
    y_train_torch = torch.tensor(y_train, dtype=torch.long)
    X_val_torch = torch.tensor(X_val, dtype=torch.float32).unsqueeze(-1)
    y_val_torch = torch.tensor(y_val, dtype=torch.long)

    train_ds = TensorDataset(X_train_torch, y_train_torch)
    val_ds = TensorDataset(X_val_torch, y_val_torch)
    train_loader = DataLoader(train_ds, batch_size=128, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=256)

    # Model
    model = TinyCNN(n_feat=X_train.shape[1], n_classes=3).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    class_weights = torch.tensor([1.0, 5.0, 10.0]).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    # Training
    model.train()
    for epoch in range(epochs):
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()

    # Validation predictions
    model.eval()
    with torch.no_grad():
        logits_val = []
        for xb, _ in val_loader:
            xb = xb.to(device)
            logits_val.append(model(xb).cpu())
        logits_val = torch.cat(logits_val, dim=0)
        P_val = torch.softmax(logits_val, dim=1).numpy()
    
    return P_val, model


def evaluate_all_models(X_train, y_train, X_val, y_val, ts_val, args):
    """Evaluate CNN and baseline models on the same train/validation split."""
    
    results = {}
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print(f"Training on {len(X_train)} samples, validating on {len(X_val)} samples")
    print(f"Using device: {device}")
    
    # 1. Train CNN (original model)
    print("\n=== Training CNN ===")
    start_time = time.time()
    cnn_probs, cnn_model = train_cnn_model(X_train, y_train, X_val, y_val, args.epochs, device)
    cnn_time = time.time() - start_time
    results['cnn'] = {
        'probs': cnn_probs,
        'training_time': cnn_time,
        'model': cnn_model
    }
    
    # 2. Train baseline models (logistic regression and rule-based)
    baseline_models = get_all_baselines_wrapped()
    
    for name, model in baseline_models.items():
        print(f"\n=== Training {name} ===")
        start_time = time.time()
        try:
            model.fit(X_train, y_train)
            probs = model.predict_proba(X_val)
            training_time = time.time() - start_time
            
            results[name] = {
                'probs': probs,
                'training_time': training_time,
                'model': model
            }
            print(f"✓ {name} trained successfully in {training_time:.2f}s")
            
        except Exception as e:
            print(f"✗ {name} failed: {e}")
            continue
    
    return results


def analyze_model_performance(results, y_val, ts_val, args):
    """Analyze and compare performance of all models."""
    
    performance_summary = []
    threshold_results = {}
    
    for model_name, result in results.items():
        probs = result['probs']
        training_time = result['training_time']
        
        # Extract class probabilities
        p_comp = probs[:, 1]  # Compression class
        p_reg = probs[:, 2]   # Regular class
        
        # Calculate AP scores
        ap_comp = average_precision_score((y_val == 1).astype(int), p_comp)
        ap_reg = average_precision_score((y_val == 2).astype(int), p_reg)
        
        # Find optimal thresholds
        best_comp = pick_threshold(
            (y_val == 1).astype(int), p_comp, ts_val,
            max_false_per_day=args.fa_day_comp
        )
        best_reg = pick_threshold(
            (y_val == 2).astype(int), p_reg, ts_val,
            max_false_per_day=args.fa_day_reg
        )
        
        # Store threshold results
        threshold_results[model_name] = {
            'comp_threshold': best_comp,
            'reg_threshold': best_reg
        }
        
        # Calculate final predictions using thresholds
        preds_val = np.zeros_like(y_val)
        for i in range(len(y_val)):
            if p_comp[i] > best_comp["T"]:
                preds_val[i] = 1
            elif p_reg[i] > best_reg["T"]:
                preds_val[i] = 2
            else:
                preds_val[i] = 0
        
        # Calculate accuracy metrics
        cm = confusion_matrix(y_val, preds_val, labels=[0, 1, 2])
        accuracy = np.trace(cm) / np.sum(cm)
        
        # Per-class metrics
        report = classification_report(y_val, preds_val, labels=[0, 1, 2], output_dict=True)
        
        # Store performance summary
        performance_summary.append({
            'model': model_name,
            'ap_compression': ap_comp,
            'ap_regular': ap_reg,
            'ap_average': (ap_comp + ap_reg) / 2,
            'accuracy': accuracy,
            'training_time': training_time,
            'comp_sensitivity': best_comp['sens'],
            'comp_fa_per_day': best_comp['fa_per_day'],
            'reg_sensitivity': best_reg['sens'], 
            'reg_fa_per_day': best_reg['fa_per_day'],
            'precision_comp': report['1']['precision'] if '1' in report else 0,
            'recall_comp': report['1']['recall'] if '1' in report else 0,
            'f1_comp': report['1']['f1-score'] if '1' in report else 0,
            'precision_reg': report['2']['precision'] if '2' in report else 0,
            'recall_reg': report['2']['recall'] if '2' in report else 0,
            'f1_reg': report['2']['f1-score'] if '2' in report else 0,
        })
        
        print(f"\n=== {model_name.upper()} RESULTS ===")
        print(f"AP Compression: {ap_comp:.3f}")
        print(f"AP Regular: {ap_reg:.3f}")
        print(f"Accuracy: {accuracy:.3f}")
        print(f"Training time: {training_time:.2f}s")
        print(f"Comp - Sens: {best_comp['sens']:.3f}, FA/day: {best_comp['fa_per_day']:.3f}")
        print(f"Reg - Sens: {best_reg['sens']:.3f}, FA/day: {best_reg['fa_per_day']:.3f}")
    
    return pd.DataFrame(performance_summary), threshold_results


def create_comparison_plots(results, y_val, save_dir):
    """Create comparison plots for all models."""
    
    save_dir = Path(save_dir)
    save_dir.mkdir(exist_ok=True)
    
    # 1. Average Precision comparison
    model_names = []
    ap_comp_scores = []
    ap_reg_scores = []
    
    for model_name, result in results.items():
        probs = result['probs']
        ap_comp = average_precision_score((y_val == 1).astype(int), probs[:, 1])
        ap_reg = average_precision_score((y_val == 2).astype(int), probs[:, 2])
        
        model_names.append(model_name)
        ap_comp_scores.append(ap_comp)
        ap_reg_scores.append(ap_reg)
    
    # Plot AP comparison
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Compression AP
    ax1.barh(model_names, ap_comp_scores)
    ax1.set_xlabel('Average Precision')
    ax1.set_title('Compression Detection - Average Precision')
    ax1.grid(True, alpha=0.3)
    
    # Regular AP  
    ax2.barh(model_names, ap_reg_scores)
    ax2.set_xlabel('Average Precision')
    ax2.set_title('Regular Low Detection - Average Precision')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_dir / 'ap_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. PR Curves for top models
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Select all available models (should be 3: CNN, logistic regression, rule-based)
    avg_ap = [(name, (ap_comp_scores[i] + ap_reg_scores[i]) / 2) 
              for i, name in enumerate(model_names)]
    top_models = sorted(avg_ap, key=lambda x: x[1], reverse=True)
    
    colors = plt.cm.Set1(np.linspace(0, 1, len(top_models)))
    
    for (model_name, _), color in zip(top_models, colors):
        probs = results[model_name]['probs']
        
        # Compression PR curve
        prec_comp, rec_comp, _ = precision_recall_curve(
            (y_val == 1).astype(int), probs[:, 1]
        )
        ap_comp = average_precision_score((y_val == 1).astype(int), probs[:, 1])
        ax1.plot(rec_comp, prec_comp, label=f'{model_name} (AP={ap_comp:.3f})', color=color)
        
        # Regular PR curve
        prec_reg, rec_reg, _ = precision_recall_curve(
            (y_val == 2).astype(int), probs[:, 2]
        )
        ap_reg = average_precision_score((y_val == 2).astype(int), probs[:, 2])
        ax2.plot(rec_reg, prec_reg, label=f'{model_name} (AP={ap_reg:.3f})', color=color)
    
    ax1.set_xlabel('Recall')
    ax1.set_ylabel('Precision')
    ax1.set_title('Compression Detection - PR Curves (All Models)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    ax2.set_xlabel('Recall')
    ax2.set_ylabel('Precision')
    ax2.set_title('Regular Low Detection - PR Curves (All Models)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_dir / 'pr_curves_all_models.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Plots saved to {save_dir}")


def main(args):
    """Main comparison function."""
    
    # Load and preprocess data (same as original)
    df = pd.read_csv(args.csv)
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

    # Build features
    X = make_features(glucose,
                      win_var=args.win_var,
                      d_short=args.delta_short,
                      d_pct=args.delta_pct)

    print(f"Dataset: {len(X)} samples, {X.shape[1]} features")
    print(f"Class distribution: {np.bincount(y)}")
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # Single train/test split for fair comparison
    if "day_id" in df.columns:
        groups = df["day_id"].astype(str).to_numpy()
        gss = GroupShuffleSplit(test_size=0.2, n_splits=1, random_state=42)
        tr, va = next(gss.split(X, y, groups))
    else:
        tr, va = train_test_split(np.arange(len(y)), test_size=0.2,
                                  stratify=y, random_state=42)

    X_train, X_val = X[tr], X[va]
    y_train, y_val = y[tr], y[va]
    ts_val = df["timestamp"].iloc[va]

    # Scale features for models that need it
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    
    print(f"Train set: {len(X_train)} samples")
    print(f"Validation set: {len(X_val)} samples")

    # Train and evaluate all models
    results = evaluate_all_models(X_train_scaled, y_train, X_val_scaled, y_val, ts_val, args)
    
    # Analyze performance
    performance_df, threshold_results = analyze_model_performance(results, y_val, ts_val, args)
    
    # Save results
    performance_df.to_csv(output_dir / 'model_comparison.csv', index=False)
    
    # Create comparison plots
    create_comparison_plots(results, y_val, output_dir)
    
    # Print summary
    print("\n" + "="*80)
    print("MODEL COMPARISON SUMMARY")
    print("="*80)
    
    # Sort by average AP
    performance_df_sorted = performance_df.sort_values('ap_average', ascending=False)
    
    print(f"{'Model':<20} {'AP Avg':<8} {'AP Comp':<8} {'AP Reg':<8} {'Accuracy':<8} {'Time(s)':<8}")
    print("-" * 80)
    
    for _, row in performance_df_sorted.iterrows():
        print(f"{row['model']:<20} {row['ap_average']:<8.3f} {row['ap_compression']:<8.3f} "
              f"{row['ap_regular']:<8.3f} {row['accuracy']:<8.3f} {row['training_time']:<8.1f}")
    
    print(f"\nDetailed results saved to: {output_dir}")
    
    # Save best model
    best_model_name = performance_df_sorted.iloc[0]['model']
    best_model = results[best_model_name]['model']
    
    # Save model bundle (similar to original script)
    bundle = {
        "scaler": scaler,
        "best_model_name": best_model_name,
        "all_results": results,
        "performance_summary": performance_df,
        "threshold_results": threshold_results,
        "feature_params": {
            "win_var": args.win_var,
            "delta_short": args.delta_short,
            "delta_pct": args.delta_pct
        }
    }
    
    joblib.dump(bundle, output_dir / 'model_comparison_bundle.joblib')
    print(f"Model bundle saved to: {output_dir / 'model_comparison_bundle.joblib'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare CNN with simple baseline models (logistic regression + rule-based)")
    parser.add_argument("--csv", required=True, help="Path to labeled CSV")
    parser.add_argument("--output_dir", default="comparison_results", help="Output directory")
    parser.add_argument("--fa_day_comp", type=float, default=0.5, help="Max false alerts/day for compression")
    parser.add_argument("--fa_day_reg", type=float, default=1.0, help="Max false alerts/day for regular")
    parser.add_argument("--win_var", type=int, default=15, help="Window (samples) for rolling variance")
    parser.add_argument("--delta_short", type=int, default=5, help="Samples for short delta (d5)")
    parser.add_argument("--delta_pct", type=int, default=10, help="Samples for %drop window")
    parser.add_argument("--epochs", type=int, default=30, help="Training epochs for CNN")
    
    args = parser.parse_args()
    main(args)