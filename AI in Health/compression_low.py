import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path


from collections import deque
from datetime import timedelta
from pathlib import Path
import shutil
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter

# =========================
# -------- CONFIG ---------
# =========================

CSV_PATH      = "Sam_Clarity.csv"
WINDOW_MIN    = 15
REBOUND_MIN   = 30
DROP_THR      = 25
REBOUND_THR   = 25
CONTAMINATION = 0.03
NIGHT_START   = 22
NIGHT_END     = 8
MERGE_GAP     = pd.Timedelta("15min")
OUT_DIR_BASE  = "plots_comp_days"
VERBOSE       = False   # set True to see internal debug prints
LOW = 80

# =========================
# ---- Utility helpers ----
# =========================
def reset_dir(path: Path):
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)

def night_mask(idx, start=NIGHT_START, end=NIGHT_END):
    return (idx.hour >= start) | (idx.hour < end)

def drop_bad_counter_rows(df, col, max_jump_factor=5):
    v = pd.to_numeric(df[col], errors='coerce')
    non_dec = v >= v.cummax()
    diffs = v.diff().abs()
    typical = diffs[diffs > 0].median()
    huge_jump = diffs > (typical * max_jump_factor)
    good = non_dec & ~huge_jump.fillna(False)
    return df.loc[good].copy(), ~good

# =========================
# ------- Loading ---------
# =========================
import re

def load_dexcom_csv(path: str, drop_watch=True, quiet=True) -> pd.DataFrame:
    raw = pd.read_csv(path, header=None, dtype=str)
    mask = raw.apply(lambda s: s.str.contains(r'^Event Type$', na=False))
    if not mask.any().any():
        raise ValueError("Could not find 'Event Type' header row.")
    header_row = mask.any(axis=1).idxmax()

    df_all = pd.read_csv(path, skiprows=header_row+1,
                         names=raw.iloc[header_row].tolist(), dtype=str)

    if drop_watch:
        device_cols = [c for c in df_all.columns if re.search(r'(device|source)', c, re.I)]
        if device_cols:
            watch_mask = df_all[device_cols].apply(
                lambda s: s.str.contains('watch', case=False, na=False)
            ).any(axis=1)
            if not quiet: print(f"Dropped {watch_mask.sum()} watch rows")
            df_all = df_all.loc[~watch_mask].copy()

    # Optional glitch filter (wrap in try in case column missing)
    counter_col = "Transmitter Time (Long Integer)"
    if counter_col in df_all.columns:
        df_all, bad_mask = drop_bad_counter_rows(df_all, counter_col)
        if not quiet: print(f"Dropped {bad_mask.sum()} counter-glitch rows")

    df = df_all[df_all["Event Type"] == "EGV"].copy()

    df["Glucose"] = (
        df["Glucose Value (mg/dL)"]
          .replace({"Low": 30, "LOW": 30, "High": 410, "HIGH": 410})
          .astype(float)
    )

    ts_col = "Timestamp (YYYY-MM-DDThh:mm:ss)"
    df["Timestamp"] = pd.to_datetime(df[ts_col])
    df = df.set_index("Timestamp").sort_index()

    out_cols = ["Glucose"]
    carb_cols    = [c for c in df.columns if "carb"    in c.lower()]
    insulin_cols = [c for c in df.columns if "insulin" in c.lower()]
    out_cols += carb_cols + insulin_cols

    df = df[out_cols]

    rename_map = {c: "carbs"   for c in df.columns if "carb"    in c.lower()}
    rename_map |= {c: "insulin" for c in df.columns if "insulin" in c.lower()}
    df = df.rename(columns=rename_map)

    for c in ("carbs", "insulin"):
        if c in df:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)

    return df

# Feature detection

def make_features(df: pd.DataFrame,
                  window_min=WINDOW_MIN,
                  rebound_min=REBOUND_MIN) -> pd.DataFrame:
    sg = df["Glucose"]
    drop    = (sg.rolling(f"{window_min}min").max() - sg).clip(lower=0)
    fwd_max = sg[::-1].rolling(f"{rebound_min}min").max()[::-1]
    rebound = (fwd_max - sg).clip(lower=0)


    # Detect flat plateaus: very low stddev in a window, and mean is hypoglycemic
    plateau_window = 20  # minutes
    plateau_std = sg.rolling(f"{plateau_window}min").std()
    # Plateau if stddev is very low, and mean is in hypoglycemic range (<= 70)
    plateau = (plateau_std < 10) & (sg.rolling(f"{plateau_window}min").mean() <= 85)

    # Flatness before and after dip
    flat_window = 20  # minutes
    flat_std_thr = 20
    # Pre-flat: stddev in window before is low
    pre_flat = sg.rolling(f"{flat_window}min").std().shift(1) < flat_std_thr
    # Post-flat: stddev in window after is low
    post_flat = sg[::-1].rolling(f"{flat_window}min").std().shift(1)[::-1] < flat_std_thr

    feat = pd.DataFrame({
        "drop": drop,
        "rebound": rebound,
        "plateau": plateau.astype(int),
        "pre_flat": pre_flat.astype(int),
        "post_flat": post_flat.astype(int)
    }, index=sg.index)

    feat["carb_sum"] = (df["carbs"].rolling(f"{window_min}min").sum().fillna(0)
                        if "carbs" in df else 0)
    feat["ins_sum"]  = (df["insulin"].rolling(f"{window_min}min").sum().fillna(0)
                        if "insulin" in df else 0)
    return feat.dropna()

def detect_compression_lows(feat: pd.DataFrame,
                            drop_thr=DROP_THR,
                            rebound_thr=REBOUND_THR,
                            contamination=CONTAMINATION) -> pd.DataFrame:
    # Use drop, plateau, pre_flat, post_flat for anomaly detection
    X = feat[["drop", "plateau", "pre_flat", "post_flat"]].values
    X_scaled = StandardScaler().fit_transform(X)
    iso = IsolationForest(n_estimators=200,
                          contamination=contamination,
                          random_state=42).fit(X_scaled)

    feat = feat.copy()
    feat["weird"] = (iso.predict(X_scaled) == -1)
    # Flag as comp_low if weird and (sharp drop+rebound) or plateau, and no carbs/insulin
    feat["comp_low"] = (
        feat["weird"] &
        (
            (
                (feat["drop"] >= drop_thr) &
                (feat["rebound"] >= rebound_thr)
            )
            |
            (feat["plateau"] == 1)
        ) &
        (feat["carb_sum"] == 0) &
        (feat["ins_sum"] == 0)
    )
    return feat

def collapse_comp_lows(feat,
                       glucose,
                       col='comp_low',
                       merge_gap=MERGE_GAP,
                       tol=5,
                       max_extend=pd.Timedelta('60min')):
    """
    Extend each compression-low episode until glucose is back within `tol`
    mg/dL of the value *before* the drop started.

    Parameters
    ----------
    feat : DataFrame  (must include boolean col `col`)
    glucose : Series  (same index as feat)
    col : str         flag column
    merge_gap : Timedelta  gap to merge flags into one episode
    tol : float       mg/dL tolerance for "back to baseline"
    max_extend : Timedelta  hard cap on how far we extend

    Returns
    -------
    feat_out : DataFrame  (with episode_id)
    episodes : DataFrame  (start, end_extended, etc.)
    """

    mask = feat[col].fillna(False)
    feat_out = feat.copy()
    feat_out['episode_id'] = pd.NA

    if not mask.any():
        return feat_out, pd.DataFrame()

    # --- initial grouping on flagged points only ---
    idx_flags = feat.index[mask]
    gaps = idx_flags.to_series().diff().gt(merge_gap).fillna(True)
    ep_ids = gaps.cumsum().astype('Int64')

    # assign initial ids to flagged rows
    feat_out.loc[idx_flags, 'episode_id'] = ep_ids.values

    records = []
    for eid in ep_ids.unique():
        seg_idx = idx_flags[ep_ids == eid]
        start_ts_core = seg_idx.min()
        end_ts_core   = seg_idx.max()

        # baseline = glucose just before the first flagged point (if exists),
        # otherwise use glucose at start.
        prev_idx = glucose.index[glucose.index < start_ts_core]
        if len(prev_idx):
            baseline_val = glucose.loc[prev_idx.max()]
        else:
            baseline_val = glucose.loc[start_ts_core]

        # search forward from the core end to find first return >= baseline_val - tol
        search_end = end_ts_core + max_extend
        forward_series = glucose.loc[end_ts_core:search_end]

        # condition for "back"
        back_mask = forward_series >= (baseline_val - tol)
        if back_mask.any():
            end_ts_ext = back_mask.idxmax()  # first True
        else:
            end_ts_ext = end_ts_core  # didn't come back within cap

        # mark ALL rows from start_ts_core to end_ts_ext with this episode_id
        span_mask = (feat_out.index >= start_ts_core) & (feat_out.index <= end_ts_ext)
        feat_out.loc[span_mask, 'episode_id'] = eid

        # collect summary
        seg_feat = feat_out.loc[span_mask]
        records.append({
            'episode_id' : eid,
            'start_core' : start_ts_core,
            'end_core'   : end_ts_core,
            'end_ext'    : end_ts_ext,
            'n_points'   : seg_feat.shape[0],
            'max_drop'   : seg_feat['drop'].max(),
            'max_rebound': seg_feat['rebound'].max(),
            'duration_min_core': (end_ts_core - start_ts_core).total_seconds()/60,
            'duration_min_ext' : (end_ts_ext  - start_ts_core).total_seconds()/60
        })

    episodes = pd.DataFrame.from_records(records).set_index('episode_id')
    return feat_out, episodes


# Plotting

def plot_days_with_comp_lows(df, feat, label,
                             out_dir_base=OUT_DIR_BASE,
                             night_only=False):
    out_dir = Path(f"{out_dir_base}_{label}")
    reset_dir(out_dir)

    mask = feat['comp_low'].fillna(False)
    if night_only:
        mask &= night_mask(feat.index)

    comp_days = feat.index[mask].normalize().unique()
    print(f"{len(comp_days)} night day(s) with compression lows.")

    saved = []
    for day in comp_days:
        day_mask      = feat.index.normalize() == day
        day_mask_flag = day_mask & mask
        if not day_mask_flag.any():
            continue

        day_gluc = df.loc[day_mask]
        day_feat = feat.loc[day_mask]

        ep_ids_today = day_feat.loc[day_mask_flag, 'episode_id'].dropna().unique()

        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(day_gluc.index, day_gluc['Glucose'], lw=1.2, label="Glucose")

        for eid in ep_ids_today:
            seg = day_feat[day_feat['episode_id'] == eid]
            start_ts, end_ts = seg.index.min(), seg.index.max()
            ax.scatter(start_ts, df.loc[start_ts, 'Glucose'],
                       s=60, facecolors='none', edgecolors='r', linewidths=2,
                       label="Compression low" if "Compression low" not in ax.get_legend_handles_labels()[1] else "")
            ax.axvspan(start_ts, end_ts, color='red', alpha=0.15)

            # Shade the 20 minutes before the compression low starts as pre-flat ONLY if flat
            preflat_start = start_ts - pd.Timedelta(minutes=20)
            preflat_end = start_ts
            preflat_chunk = df.loc[preflat_start:preflat_end]["Glucose"]
            flat_std_thr = 20  # must match feature extraction
            if len(preflat_chunk) > 1 and preflat_chunk.std() < flat_std_thr:
                ax.axvspan(preflat_start, preflat_end, color='blue', alpha=0.10, zorder=0, label='Pre-flat' if 'Pre-flat' not in ax.get_legend_handles_labels()[1] else "")

        y = day_gluc['Glucose']
        ymin = max(0, y.min() - 10)
        ymax = y.max() + 10
        ax.set_ylim(30, 410)
        ax.set_title(f"{label} – {day.date()} (Compression Lows)")
        ax.set_ylabel("mg/dL")
        ax.set_xlabel("Time")
        ax.xaxis.set_major_formatter(DateFormatter("%H:%M"))
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
        
        # Target range shading
        ax.axhspan(30, 70,  color="#f05e5e", alpha=0.10, zorder=0)   # low zone (soft red)
        ax.axhspan(180, 410, color="#edc55f", alpha=0.12, zorder=0)   # high zone (muted yellow)
        
        plt.tight_layout()

        fn = out_dir / f"{label}_comp_day_{day.date()}.png"
        plt.savefig(fn, dpi=200, bbox_inches='tight')
        plt.close(fig)
        saved.append(str(fn))

    if saved:
        print(f"Plots saved to: {out_dir.resolve()}")
    return saved

# =========================
# ---------- MAIN ---------
# =========================
if __name__ == "__main__":
    df    = load_dexcom_csv(CSV_PATH, quiet=not VERBOSE)
    label = Path(CSV_PATH).stem

    feat  = make_features(df)
    feat  = detect_compression_lows(feat)
    feat, episodes = collapse_comp_lows(
    feat,
    glucose=df['Glucose'],          # <— add this
    col='comp_low',
    merge_gap=MERGE_GAP,
    tol=5,
    max_extend=pd.Timedelta('60min')
    )

    # --- Night-only view ---
    nm = night_mask(feat.index)
    night_events = feat['comp_low'] & nm

    # --- Print concise summary ---
    print("\n=== Compression Low Summary ===")
    print(f"Total points:      {len(feat):,}")
    print(f"Total episodes:    {episodes.shape[0]}")
    print(f"Night episodes:    {feat.loc[night_events, 'episode_id'].nunique()}")
    print(f"Total flags:       {feat['comp_low'].sum()}")
    print(f"Night flags:       {night_events.sum()}")

    night_days = feat.index[night_events].normalize().unique()
    print("Night days:", list(night_days))

    # --- Plots (night only) ---
    plot_days_with_comp_lows(df, feat, label, night_only=True)




## % New section

import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# ---------- 1. Collapse consecutive flags into episodes ----------
def collapse_events(mask: pd.Series, gap_minutes=10) -> pd.DataFrame:
    idx = mask.index[mask]
    if idx.empty:
        return pd.DataFrame(columns=["start", "end"])
    gaps = idx.to_series().diff().gt(pd.Timedelta(minutes=gap_minutes))
    groups = gaps.cumsum()
    out = idx.to_series().groupby(groups).agg(["min", "max"])
    out.columns = ["start", "end"]
    return out.reset_index(drop=True)

# ---------- 2. Interactive reviewer ----------
def review_episodes_plot_only(
    df: pd.DataFrame,
    episodes: pd.DataFrame,
    glucose_col: str = "Glucose",
    pre_minutes: int = 60,
    post_minutes: int = 60,
    save_csv: str | None = "episode_labels.csv"
):
    """
    Plot each episode with +/- pre/post minutes for quick visual review.
    Tag: y(correct) / n(incorrect) / s(skip) / q(quit).
    Closes plot after input.
    """
    labels_path = Path(save_csv) if save_csv else None
    if labels_path and labels_path.exists():
        labels = pd.read_csv(labels_path, parse_dates=["start", "end"])
    else:
        labels = pd.DataFrame(columns=["start", "end", "label", "notes"])

    # Find which episodes still need review
    done_pairs = set(zip(labels["start"], labels["end"]))
    todo = episodes[~episodes.apply(lambda r: (r["start"], r["end"]) in done_pairs, axis=1)]

    if todo.empty:
        print("No episodes to review.")
        return labels

    for i, row in todo.iterrows():
        start, end = row["start"], row["end"]
        win_start = start - pd.Timedelta(minutes=pre_minutes)
        win_end = end + pd.Timedelta(minutes=post_minutes)
        chunk = df.loc[win_start:win_end]

        plt.figure(figsize=(10, 4))
        plt.plot(chunk.index, chunk[glucose_col].astype(float), linewidth=2)
        plt.axvspan(start, end, alpha=0.25, color="red")
        plt.title(f"Episode {i+1}/{len(todo)}  |  {start} → {end}")
        plt.xlabel("Time")
        plt.ylabel("Glucose (mg/dL)")
        plt.tight_layout()
        plt.show(block=False)

        ans = input("y=correct, n=incorrect, s=skip, q=quit: ").strip().lower()
        plt.close()

        if ans == "q":
            print("Quitting, saving progress…")
            break
        if ans not in {"y", "n"}:
            continue

        note = input("Optional note (enter to skip): ").strip()
        label_val = "correct" if ans == "y" else "incorrect"

        new_label = {"start": start, "end": end, "label": label_val, "notes": note}
        labels = pd.concat([
            labels,
            pd.DataFrame([new_label])
        ], ignore_index=True)

        if labels_path:
            labels.to_csv(labels_path, index=False)
            print("Saved.")

    return labels



