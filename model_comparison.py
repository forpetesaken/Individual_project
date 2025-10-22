"""
Model Comparison Script
======================

Compare CNN model against simple baseline approaches:
- Logistic Regression (statistical baseline)

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
from AI_in_Health.compression_onset import make_features, pick_threshold, TinyCNN
from baseline_models import get_all_baselines_wrapped



def load_pretrained_cnn_model(df_val, y_val, cnn_bundle_path):
    """Load the pre-trained CNN model and get predictions using the same approach as compression_onset_test."""
    
    # Load the model bundle
    bundle = joblib.load(cnn_bundle_path)
    
    # Get feature parameters from bundle (same as compression_onset_test)
    win_var = bundle.get("win_var", 15)
    d_short = bundle.get("delta_short", 5) 
    d_pct = bundle.get("delta_pct", 10)
    
    print(f"Using CNN bundle feature params: win_var={win_var}, d_short={d_short}, d_pct={d_pct}")
    
    # Build features using the SAME parameters as the trained model
    glucose_val = df_val["glucose"].to_numpy(dtype=float)
    X_val = make_features(glucose_val, win_var, d_short, d_pct)
    
    # Use the CNN's own scaler from the bundle
    scaler = bundle["scaler"]
    X_val_scaled = scaler.transform(X_val)
    
    # Setup device and model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Reshape for CNN (add sequence dimension)
    X_val_torch = torch.tensor(X_val_scaled, dtype=torch.float32).unsqueeze(-1)
    
    # Create and load model
    model = TinyCNN(n_feat=X_val.shape[1], n_classes=3).to(device)
    model.load_state_dict(bundle["model_state_dict"])
    model.eval()
    
    # Get predictions (same as compression_onset_test)
    with torch.no_grad():
        logits = model(X_val_torch.to(device)).cpu()
        P_val = torch.softmax(logits, dim=1).numpy()
    
    return P_val, model, scaler


def evaluate_all_models(df_train, df_val, y_train, y_val, ts_val, args):
    """Evaluate CNN and baseline models on the same train/validation split."""
    
    results = {}
    
    print(f"Training on {len(df_train)} samples, validating on {len(df_val)} samples")
    
    # 1. Load pre-trained CNN model (uses its own feature extraction)
    print("\n=== Loading Pre-trained CNN ===")
    start_time = time.time()
    cnn_probs, cnn_model, cnn_scaler = load_pretrained_cnn_model(df_val, y_val, args.cnn_bundle)
    cnn_time = time.time() - start_time
    results['cnn'] = {
        'probs': cnn_probs,
        'training_time': cnn_time,
        'model': cnn_model,
        'scaler': cnn_scaler
    }
    
    # 2. For baseline models, we need to build features using comparison script parameters
    glucose_train = df_train["glucose"].to_numpy(dtype=float)
    glucose_val = df_val["glucose"].to_numpy(dtype=float)
    
    X_train = make_features(glucose_train,
                           win_var=args.win_var,
                           d_short=args.delta_short,
                           d_pct=args.delta_pct)
    X_val = make_features(glucose_val,
                         win_var=args.win_var,
                         d_short=args.delta_short,
                         d_pct=args.delta_pct)
    
    # Scale features for baseline models
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    
    # Train baseline models (logistic regression)
    baseline_models = get_all_baselines_wrapped()
    
    for name, model in baseline_models.items():
        print(f"\n=== Training {name} ===")
        start_time = time.time()
        try:
            model.fit(X_train_scaled, y_train)
            probs = model.predict_proba(X_val_scaled)
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


def create_comparison_table_and_confusion_matrices(results, y_val, threshold_results, save_dir):
    """Create a comparison table and confusion matrix figures for all models."""
    
    save_dir = Path(save_dir)
    save_dir.mkdir(exist_ok=True)
    
    # Prepare data for comparison table
    model_data = []
    
    for model_name, result in results.items():
        probs = result['probs']
        
        # Calculate metrics
        ap_comp = average_precision_score((y_val == 1).astype(int), probs[:, 1])
        ap_reg = average_precision_score((y_val == 2).astype(int), probs[:, 2])
        
        # Get thresholds
        best_comp = threshold_results[model_name]['comp_threshold']
        best_reg = threshold_results[model_name]['reg_threshold']
        
        # Calculate predictions using thresholds
        preds_val = np.zeros_like(y_val)
        for i in range(len(y_val)):
            if probs[i, 1] > best_comp["T"]:
                preds_val[i] = 1
            elif probs[i, 2] > best_reg["T"]:
                preds_val[i] = 2
            else:
                preds_val[i] = 0
        
        # Calculate confusion matrix and metrics
        cm = confusion_matrix(y_val, preds_val, labels=[0, 1, 2])
        accuracy = np.trace(cm) / np.sum(cm)
        
        # Per-class metrics with proper handling
        report = classification_report(y_val, preds_val, labels=[0, 1, 2], output_dict=True)
        
        comp_precision = 0.0
        comp_f1 = 0.0
        reg_precision = 0.0
        reg_f1 = 0.0
        
        if '1' in report and isinstance(report['1'], dict):
            comp_precision = report['1'].get('precision', 0.0)
            comp_f1 = report['1'].get('f1-score', 0.0)
        
        if '2' in report and isinstance(report['2'], dict):
            reg_precision = report['2'].get('precision', 0.0)
            reg_f1 = report['2'].get('f1-score', 0.0)
        
        model_data.append({
            'Model': model_name.upper(),
            'Accuracy': f"{accuracy:.3f}",
            'Compression Precision': f"{comp_precision:.3f}",
            'Compression F1-Score': f"{comp_f1:.3f}",
            'Regular Precision': f"{reg_precision:.3f}",
            'Regular F1-Score': f"{reg_f1:.3f}"
        })
    
    # Create comparison table figure
    fig, ax = plt.subplots(figsize=(20, 8))
    ax.axis('tight')
    ax.axis('off')
    
    # Convert to DataFrame for easier handling
    table_df = pd.DataFrame(model_data)
    
    # Create table
    table = ax.table(cellText=table_df.values.tolist(),
                     colLabels=table_df.columns.tolist(),
                     cellLoc='center',
                     loc='center')
    
    # Style the table
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 2)
    
    # Color header row
    for i in range(len(table_df.columns)):
        table[(0, i)].set_facecolor('#4CAF50')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    # Alternate row colors
    for i in range(1, len(table_df) + 1):
        for j in range(len(table_df.columns)):
            if i % 2 == 0:
                table[(i, j)].set_facecolor('#f0f0f0')
    
    plt.title('Model Comparison Table: CNN vs Baseline Model', 
              fontsize=16, fontweight='bold', pad=20)
    plt.savefig(save_dir / 'model_comparison_table.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Create confusion matrices for each model
    n_models = len(results)
    fig, axes = plt.subplots(1, n_models, figsize=(6 * n_models, 5))
    
    if n_models == 1:
        axes = [axes]
    
    for idx, (model_name, result) in enumerate(results.items()):
        probs = result['probs']
        
        # Get thresholds
        best_comp = threshold_results[model_name]['comp_threshold']
        best_reg = threshold_results[model_name]['reg_threshold']
        
        # Calculate predictions using thresholds
        preds_val = np.zeros_like(y_val)
        for i in range(len(y_val)):
            if probs[i, 1] > best_comp["T"]:
                preds_val[i] = 1
            elif probs[i, 2] > best_reg["T"]:
                preds_val[i] = 2
            else:
                preds_val[i] = 0
        
        # Calculate confusion matrix
        cm = confusion_matrix(y_val, preds_val, labels=[0, 1, 2])
        
        # Plot confusion matrix
        ax = axes[idx]
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
        ax.set_title(f'{model_name.upper()}\nConfusion Matrix', fontsize=14, fontweight='bold')
        ax.set_xticks([0, 1, 2])
        ax.set_yticks([0, 1, 2])
        ax.set_xticklabels(['None', 'Compression', 'Regular'])
        ax.set_yticklabels(['None', 'Compression', 'Regular'])
        
        # Add accuracy text
        accuracy = np.trace(cm) / np.sum(cm)
        ax.text(0.5, -0.15, f'Accuracy: {accuracy:.3f}', 
                transform=ax.transAxes, ha='center', fontsize=11, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(save_dir / 'confusion_matrices.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Comparison table and confusion matrices saved to {save_dir}")


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

    print(f"Dataset: {len(df)} samples")
    print(f"Class distribution: {np.bincount(y)}")
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # Single train/test split for fair comparison
    if "day_id" in df.columns:
        groups = df["day_id"].astype(str).to_numpy()
        gss = GroupShuffleSplit(test_size=0.2, n_splits=1, random_state=42)
        tr, va = next(gss.split(df, y, groups))
    else:
        tr, va = train_test_split(np.arange(len(y)), test_size=0.2,
                                  stratify=y, random_state=42)

    df_train, df_val = df.iloc[tr].copy(), df.iloc[va].copy()
    y_train, y_val = y[tr], y[va]
    ts_val = df["timestamp"].iloc[va]
    
    print(f"Train set: {len(df_train)} samples")
    print(f"Validation set: {len(df_val)} samples")

    # Train and evaluate all models
    results = evaluate_all_models(df_train, df_val, y_train, y_val, ts_val, args)
    
    # Analyze performance
    performance_df, threshold_results = analyze_model_performance(results, y_val, ts_val, args)
    
    # Save results
    performance_df.to_csv(output_dir / 'model_comparison.csv', index=False)
    
    # Create comparison plots
    create_comparison_table_and_confusion_matrices(results, y_val, threshold_results, output_dir)
    
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
    # Use the scaler from the best baseline model if available
    best_model_result = results[best_model_name]
    baseline_scaler = StandardScaler()  # Create a new scaler as fallback
    
    bundle = {
        "baseline_scaler": baseline_scaler,
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
    parser = argparse.ArgumentParser(description="Compare CNN with logistic regression baseline model")
    parser.add_argument("--csv", required=True, help="Path to labeled CSV")
    parser.add_argument("--cnn_bundle", required=True, help="Path to pre-trained CNN model bundle")
    parser.add_argument("--output_dir", default="comparison_results", help="Output directory")
    parser.add_argument("--fa_day_comp", type=float, default=0.5, help="Max false alerts/day for compression")
    parser.add_argument("--fa_day_reg", type=float, default=1.0, help="Max false alerts/day for regular")
    parser.add_argument("--win_var", type=int, default=15, help="Window (samples) for rolling variance")
    parser.add_argument("--delta_short", type=int, default=5, help="Samples for short delta (d5)")
    parser.add_argument("--delta_pct", type=int, default=10, help="Samples for %drop window")
    
    args = parser.parse_args()
    main(args)