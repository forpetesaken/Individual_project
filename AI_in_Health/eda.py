#!/usr/bin/env python3

import argparse
import os
from pathlib import Path
import textwrap

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

LOW_THRESH = 70.0  # mg/dL threshold for hypoglycemia shading

def read_data(csv_path: str) -> pd.DataFrame:
    """Read the CSV and normalize columns."""
    df = pd.read_csv(csv_path)

    # Normalize column names
    df.columns = [c.strip().lower() for c in df.columns]
    # Try common variants
    time_col_candidates = ["timestamp", "time", "datetime", "date", "dt"]
    glucose_col_candidates = ["glucose", "bg", "value", "cgmg", "glucose_mgdl"]
    label_col_candidates = ["label", "y", "target", "event"]

    tcol = next((c for c in time_col_candidates if c in df.columns), None)
    gcol = next((c for c in glucose_col_candidates if c in df.columns), None)
    lcol = next((c for c in label_col_candidates if c in df.columns), None)

    if gcol is None:
        raise ValueError("Could not find a glucose column. Looked for: " + ", ".join(glucose_col_candidates))
    if tcol is None:
        # Allow missing timestamps; we can still do distributional EDA
        df["timestamp"] = pd.RangeIndex(start=0, stop=len(df), step=1)
        tcol = "timestamp"
    else:
        # Parse datetimes if possible
        df[tcol] = pd.to_datetime(df[tcol], errors="coerce")

    # Rename to canonical
    if tcol != "timestamp":
        df = df.rename(columns={tcol: "timestamp"})
    if gcol != "glucose":
        df = df.rename(columns={gcol: "glucose"})
    if lcol and lcol != "label":
        df = df.rename(columns={lcol: "label"})

    # Coerce glucose to numeric
    df["glucose"] = pd.to_numeric(df["glucose"], errors="coerce")

    # Sort by time if time is datetime-like
    if pd.api.types.is_datetime64_any_dtype(df["timestamp"]):
        df = df.sort_values("timestamp").reset_index(drop=True)

    return df


def infer_sampling_minutes(ts: pd.Series) -> float | None:
    """Infer typical sampling interval in minutes from a datetime series."""
    if not pd.api.types.is_datetime64_any_dtype(ts):
        return None
    deltas = ts.diff().dropna().dt.total_seconds() / 60.0
    if len(deltas) == 0:
        return None
    return float(np.median(deltas))


def basic_stats(x: pd.Series) -> dict:
    """Compute basic descriptive stats on a numeric Series."""
    x_clean = x.dropna()
    quantiles = x_clean.quantile([0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99])
    stats = {
        "count": int(x_clean.count()),
        "missing": int(x.isna().sum()),
        "mean": float(x_clean.mean()),
        "median": float(x_clean.median()),
        "std": float(x_clean.std(ddof=1)) if x_clean.size > 1 else np.nan,
        "var": float(x_clean.var(ddof=1)) if x_clean.size > 1 else np.nan,
        "min": float(x_clean.min()),
        "max": float(x_clean.max()),
        "skew": float(x_clean.skew()),
        "kurtosis": float(x_clean.kurtosis()),
        "q01": float(quantiles.loc[0.01]),
        "q05": float(quantiles.loc[0.05]),
        "q25": float(quantiles.loc[0.25]),
        "q50": float(quantiles.loc[0.50]),
        "q75": float(quantiles.loc[0.75]),
        "q95": float(quantiles.loc[0.95]),
        "q99": float(quantiles.loc[0.99]),
        "iqr": float(quantiles.loc[0.75] - quantiles.loc[0.25]),
    }
    return stats


def iqr_outliers(x: pd.Series, k: float = 1.5) -> pd.Series:
    """Return boolean mask for IQR-based outliers."""
    x_clean = x.dropna()
    if x_clean.empty:
        return pd.Series(False, index=x.index)
    q1, q3 = x_clean.quantile([0.25, 0.75])
    iqr = q3 - q1
    lower = q1 - k * iqr
    upper = q3 + k * iqr
    return (x < lower) | (x > upper)


def acf(x: np.ndarray, nlags: int = 60) -> np.ndarray:
    """
    Simple (biased) autocorrelation function up to nlags.
    Returns array of length nlags+1 where acf[0] == 1.
    """
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    if x.size == 0:
        return np.array([np.nan] * (nlags + 1))
    x = x - x.mean()
    denom = np.dot(x, x)
    out = [1.0]
    for lag in range(1, nlags + 1):
        if lag >= x.size:
            out.append(np.nan)
        else:
            num = np.dot(x[:-lag], x[lag:])
            out.append(num / denom if denom != 0 else np.nan)
    return np.array(out)


def _add_note(text: str, loc: str = "upper left"):
    """Add a small explanatory note box inside the current axes."""
    loc_map = {
        "upper left": (0.02, 0.98, "top"),
        "upper right": (0.98, 0.98, "top"),
        "lower left": (0.02, 0.02, "bottom"),
        "lower right": (0.98, 0.02, "bottom"),
    }
    x, y, va = loc_map.get(loc, (0.02, 0.98, "top"))
    plt.gca().text(
        x, y,
        textwrap.fill(text, 70),
        transform=plt.gca().transAxes,
        va=va, ha="left",
        fontsize=9,
        bbox=dict(boxstyle="round,pad=0.35", fc="white", ec="black", alpha=0.8)
    )


def make_plots(df: pd.DataFrame, outdir: Path, roll_minutes: int | None):
    outdir.mkdir(parents=True, exist_ok=True)
    x = df["glucose"]
    below = x < LOW_THRESH
    pct_below = below.mean() if x.count() else 0.0

    # 1) Time series
    plt.figure()
    ax = plt.gca()
    if pd.api.types.is_datetime64_any_dtype(df["timestamp"]):
        t = df["timestamp"]
        plt.plot(t, x, lw=1)
        # Shade below-threshold points by vertical spans between consecutive timestamps
        if t.notna().sum() > 1:
            # Mark below-threshold segments
            for i in range(1, len(df)):
                if pd.notna(t.iloc[i-1]) and pd.notna(t.iloc[i]):
                    if bool(below.iloc[i-1]) or bool(below.iloc[i]):
                        ax.fill_between([t.iloc[i-1], t.iloc[i]],
                                        [x.iloc[i-1], x.iloc[i]],
                                        [LOW_THRESH, LOW_THRESH],
                                        where=[True, True],
                                        color="red", alpha=0.15, step="pre")
        plt.xlabel("Time")
    else:
        idx = np.arange(len(x))
        plt.plot(idx, x, lw=1)
        # Shade below threshold segments by index
        if len(idx) > 1:
            for i in range(1, len(idx)):
                if bool(below.iloc[i-1]) or bool(below.iloc[i]):
                    ax.fill_between([idx[i-1], idx[i]],
                                    [x.iloc[i-1], x.iloc[i]],
                                    [LOW_THRESH, LOW_THRESH],
                                    color="red", alpha=0.15, step="pre")
        plt.xlabel("Index")
    plt.axhline(LOW_THRESH, linestyle="--")
    plt.ylabel("Glucose (mg/dL)")
    plt.title("Glucose Time Series")
    _add_note(
        f"Line shows glucose over time. Dashed line is 70 mg/dL.\n"
        f"Red shading highlights values below 70 (≈ {pct_below:.1%} of readings)."
    )
    plt.tight_layout()
    plt.savefig(outdir / "timeseries.png", dpi=160)
    plt.close()

    # 2) Rolling mean (if we can infer a cadence)
    if roll_minutes and roll_minutes > 0:
        window_pts = max(1, int(round(60 / roll_minutes)))  # ~1 hour window
        roll = x.rolling(window_pts, min_periods=max(1, window_pts // 3)).mean()

        plt.figure()
        ax = plt.gca()
        if pd.api.types.is_datetime64_any_dtype(df["timestamp"]):
            t = df["timestamp"]
            plt.plot(t, x, alpha=0.5, lw=1, label="glucose")
            plt.plot(t, roll, lw=2, label=f"rolling mean (~{60} min)")
            # shade below threshold
            if t.notna().sum() > 1:
                for i in range(1, len(df)):
                    if pd.notna(t.iloc[i-1]) and pd.notna(t.iloc[i]):
                        if bool(below.iloc[i-1]) or bool(below.iloc[i]):
                            ax.fill_between([t.iloc[i-1], t.iloc[i]],
                                            [x.iloc[i-1], x.iloc[i]],
                                            [LOW_THRESH, LOW_THRESH],
                                            color="red", alpha=0.15, step="pre")
            plt.xlabel("Time")
        else:
            idx = np.arange(len(x))
            plt.plot(idx, x, alpha=0.5, lw=1, label="glucose")
            plt.plot(idx, roll, lw=2, label=f"rolling mean (~{60} min)")
            for i in range(1, len(idx)):
                if bool(below.iloc[i-1]) or bool(below.iloc[i]):
                    ax.fill_between([idx[i-1], idx[i]],
                                    [x.iloc[i-1], x.iloc[i]],
                                    [LOW_THRESH, LOW_THRESH],
                                    color="red", alpha=0.15, step="pre")
            plt.xlabel("Index")

        plt.axhline(LOW_THRESH, linestyle="--")
        plt.ylabel("Glucose (mg/dL)")
        plt.title("Glucose with Rolling Mean")
        plt.legend()
        _add_note(
            "Rolling mean (~1 hour) smooths short-term noise to reveal trends.\n"
            "Red shading marks readings below 70 mg/dL; use to spot low clusters."
        )
        plt.tight_layout()
        plt.savefig(outdir / "rolling_mean.png", dpi=160)
        plt.close()

    # 3) Histogram
    plt.figure()
    counts, bins, patches = plt.hist(x.dropna(), bins=30)
    # Color bins whose centers are below threshold
    for b, p in zip(bins[:-1], patches):
        center = (b + (b + (bins[1]-bins[0]))) / 2.0  # rough center
        # Better center: b + (next_b - b)/2 — compute safely
    plt.cla()
    # Re-do with control over bar colors
    values = x.dropna().to_numpy()
    counts, bin_edges = np.histogram(values, bins=30)
    centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0
    widths = np.diff(bin_edges)
    for c, w, h in zip(centers, widths, counts):
        color = "red" if c < LOW_THRESH else None
        plt.bar(c, h, width=w, align="center", color=color)
    plt.axvline(LOW_THRESH, linestyle="--")
    plt.xlabel("Glucose (mg/dL)")
    plt.ylabel("Count")
    plt.title("Glucose Distribution")
    _add_note(
        "Histogram shows overall distribution. Bars left of 70 mg/dL are red.\n"
        "Use to assess central tendency, spread, and prevalence of low values."
    )
    plt.tight_layout()
    plt.savefig(outdir / "histogram.png", dpi=160)
    plt.close()

    # 4) Boxplot
    plt.figure()
    bp = plt.boxplot(x.dropna(), vert=True, whis=1.5, showmeans=True)
    plt.axhline(LOW_THRESH, linestyle="--")
    plt.ylabel("Glucose (mg/dL)")
    plt.title("Glucose Boxplot (IQR whiskers)")
    _add_note(
        "Box shows IQR (Q1–Q3), line is median; whiskers reach 1.5×IQR.\n"
        "Points beyond are outliers. Dashed line marks 70 mg/dL threshold."
    )
    plt.tight_layout()
    plt.savefig(outdir / "boxplot.png", dpi=160)
    plt.close()

    # 5) Autocorrelation (first 60 lags)
    acf_vals = acf(x.to_numpy(), nlags=60)
    plt.figure()
    lags = np.arange(acf_vals.size)
    try:
        # Older Matplotlib (<3.8) accepted use_line_collection
        plt.stem(lags, acf_vals, use_line_collection=True)
    except TypeError:
        plt.stem(lags, acf_vals)
    plt.xlabel("Lag")
    plt.ylabel("ACF")
    plt.title("Glucose Autocorrelation (up to 60 lags)")
    _add_note(
        "Autocorrelation indicates how current readings relate to prior ones.\n"
        "Slowly decaying ACF suggests strong persistence; spikes imply periodicity."
    )
    plt.tight_layout()
    plt.savefig(outdir / "autocorrelation.png", dpi=160)
    plt.close()


def make_markdown_report(stats: dict, df: pd.DataFrame, outdir: Path, cadence_min: float | None,
                         outlier_mask: pd.Series):
    """Write a tiny Markdown file that summarizes key findings and links plots."""
    label_info = ""
    if "label" in df.columns:
        label_counts = df["label"].value_counts(dropna=False).to_dict()
        label_info = "\n".join([f"- label={k}: {v}" for k, v in label_counts.items()])

    time_coverage = ""
    if pd.api.types.is_datetime64_any_dtype(df["timestamp"]):
        start = df["timestamp"].min()
        end = df["timestamp"].max()
        time_coverage = f"{start} → {end} (span: {end - start})"

    md = f"""# CGM EDA Summary

**Rows:** {len(df)}  
**Glucose non-missing:** {stats['count']} (missing: {stats['missing']})  
**Time coverage:** {time_coverage or 'N/A'}  
**Inferred sampling cadence:** {cadence_min:.2f} min per reading if detected, else N/A

## Glucose Stats
- mean: {stats['mean']:.3f}
- median: {stats['median']:.3f}
- variance: {stats['var']:.3f}
- std dev: {stats['std']:.3f}
- min/max: {stats['min']:.3f} / {stats['max']:.3f}
- skew: {stats['skew']:.3f}
- kurtosis: {stats['kurtosis']:.3f}
- quartiles (Q1/median/Q3): {stats['q25']:.3f} / {stats['q50']:.3f} / {stats['q75']:.3f}
- IQR: {stats['iqr']:.3f}
- p01 / p05 / p95 / p99: {stats['q01']:.3f} / {stats['q05']:.3f} / {stats['q95']:.3f} / {stats['q99']:.3f}

## IQR Outliers
- count: {int(outlier_mask.sum())}
- proportion: {outlier_mask.mean():.3%}

## Label Distribution
{label_info or '(no label column or all missing)'}

## Figures
- Time series: `timeseries.png`
- Rolling mean: `rolling_mean.png` (if cadence inferred)
- Histogram: `histogram.png`
- Boxplot: `boxplot.png`
- Autocorrelation: `autocorrelation.png`
"""
    (outdir / "SUMMARY.md").write_text(textwrap.dedent(md), encoding="utf-8")


def main():
    ap = argparse.ArgumentParser(description="Exploratory data analysis for CGM glucose CSV.")
    ap.add_argument("--csv", required=True, help="Path to CSV (expects 'timestamp,glucose[,label]')")
    ap.add_argument("--outdir", default="./eda_out", help="Directory to write outputs")
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    df = read_data(args.csv)

    # Infer sampling cadence
    cadence_min = infer_sampling_minutes(df["timestamp"])

    # Core stats
    stats = basic_stats(df["glucose"])

    # Outliers
    out_mask = iqr_outliers(df["glucose"], k=1.5)

    # Save stats as CSV for quick copy/paste
    pd.DataFrame([stats]).to_csv(outdir / "glucose_stats.csv", index=False)

    # Plots (filenames unchanged)
    make_plots(df, outdir, roll_minutes=int(round(cadence_min)) if cadence_min else None)

    # Markdown report
    make_markdown_report(stats, df, outdir, cadence_min, out_mask)

    # Console summary
    print("=== CGM EDA ===")
    print(f"Rows: {len(df)}  |  Glucose (non-missing): {stats['count']}  |  Missing: {stats['missing']}")
    if pd.api.types.is_datetime64_any_dtype(df["timestamp"]):
        print(f"Time coverage: {df['timestamp'].min()} -> {df['timestamp'].max()} "
              f"(span: {df['timestamp'].max() - df['timestamp'].min()})")
    if cadence_min:
        print(f"Inferred sampling cadence: ~{cadence_min:.2f} minutes")
    print("\nGlucose summary:")
    for k in ["mean","median","var","std","min","max","skew","kurtosis","q01","q05","q25","q50","q75","q95","q99","iqr"]:
        print(f"  {k:>9}: {stats[k]:.4f}")
    print(f"\nIQR outliers: {int(out_mask.sum())} ({out_mask.mean():.2%})")
    if "label" in df.columns:
        print("\nLabel counts:")
        print(df["label"].value_counts(dropna=False))
    print(f"\nWrote outputs to: {outdir.resolve()}")
    print(" Files:")
    for p in sorted(outdir.glob("*")):
        print("  -", p.name)


if __name__ == "__main__":
    main()
