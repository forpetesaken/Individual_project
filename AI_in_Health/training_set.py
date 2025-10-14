import numpy as np, pandas as pd
# Import numpy and pandas libraries for data manipulation

# ---------- 0.  Load point‑level labels ----------
import pandas as pd
import numpy as np

# --- 1.1  read your CGM file (rename columns to suit) ---
df_raw = pd.read_csv("AI in Health/sample1.csv", parse_dates=["timestamp"])
df_raw = df_raw.set_index("timestamp")
df_raw = df_raw.sort_index()  # Ensure index is sorted for asof()
df_raw["glucose"] = pd.to_numeric(df_raw["glucose"], errors="coerce")  # Ensure glucose is float
gluc    = df_raw["glucose"].to_numpy()         # numpy float array

# --- 1.2  call your feature function ---

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

feat_arr = make_features(gluc,
                         win_var = 15,   # 15 min variance window
                         d_short = 5,    # 5‑min delta
                         d_pct   = 10)   # 10‑min %drop

# --- 1.3  wrap back into a dataframe with the same index ---
feat_cols = ["g", "d1", "d5", "accel", "var15", "pct10"]
feat = pd.DataFrame(feat_arr, index=df_raw.index, columns=feat_cols)

# ---------- 1.  Collapse to episodes ----------
def collapse(mask, gap='10min'):
    # Collapse consecutive True values in a boolean mask into episodes, separated by gaps longer than 'gap'
    idx = mask.index[mask]
    # Get the index values where the mask is True
    gaps = idx.to_series().diff().gt(pd.Timedelta(gap)).fillna(True)
    # Find gaps between True indices greater than the specified gap duration
    return idx.to_series().groupby(gaps.cumsum()).agg(["min", "max"])
    # Group by gap segments and aggregate to get the start ('min') and end ('max') of each episode

comp_episodes     = collapse(df_raw['label'] == 1)   # compression
# Find episodes where the label is 1 (compression)
reg_hypo_episodes = collapse(df_raw['label'] == 2)   # regular
# Find episodes where the label is 2 (regular hypoglycemia)

comp_episodes['etype']     = 1
# Assign episode type 1 to compression episodes
reg_hypo_episodes['etype'] = 2
# Assign episode type 2 to regular hypoglycemia episodes

episodes = (pd.concat([comp_episodes, reg_hypo_episodes])
            # Concatenate compression and regular episodes into one DataFrame
              .rename(columns={'min':'start_core', 'max':'end_core'})
            # Rename columns to 'start_core' and 'end_core'
              .reset_index(drop=True))
            # Reset the index for the combined episodes DataFrame

# ---------- 2.  Build your *feature* dataframe ----------
# (Replace this with your real feature‑extraction pipeline)
feat = df_raw.drop(columns=['label'])          # <-- remove label so it can’t leak
# Remove the 'label' column from the raw DataFrame to prevent label leakage into features
# ... plus any engineered columns like drop, plateau, etc.
# (Add any additional engineered feature columns here as needed)

# ---------- 3.  Parameters ----------
CTX_MIN   = 120          # 2 h context
# Context window size in minutes (2 hours)
STEP_MIN  = 5            # CGM cadence
# Step size in minutes (CGM measurement interval)
WIN_STEPS = CTX_MIN // STEP_MIN   # 24
# Number of steps in the window (context divided by step size)

rng     = np.random.default_rng(42)
# Create a random number generator with a fixed seed for reproducibility
offset  = pd.Timedelta(minutes=CTX_MIN)
# Offset as a timedelta object representing the context window

# ---------- 4.  Helper to flatten one window ----------
def flatten(chunk, label):
    # Convert a chunk of data (24 time steps × N features) into a flat dictionary for ML
    """24×Nfeat -> 1 flat dict"""
    row = {}
    # Initialize an empty dictionary to hold flattened features
    for i, (_, vals) in enumerate(chunk.iterrows()):
        # Iterate over each row in the chunk
        step = -WIN_STEPS + i + 1      # -23 … 0
        # Calculate the relative time step (from -23 to 0)
        for col in chunk.columns:
            # For each feature column
            row[f"{col}_t{step}"] = vals[col]
            # Store the value with a key indicating feature and time step
    row['label'] = label
    # Add the label for this window
    return row
    # Return the flattened dictionary

# ---------- 5.  Positive windows ----------
rows = []
for _, ep in episodes.iterrows():
    win_start = ep['start_core'] - offset
    # Use asof to get nearest available timestamps
    start_idx = feat.index.asof(win_start)
    end_idx = feat.index.asof(ep['start_core'])
    if pd.isna(start_idx) or pd.isna(end_idx):
        continue  # skip if no valid index
    chunk = feat.loc[start_idx:end_idx].tail(WIN_STEPS)
    if len(chunk) == WIN_STEPS:
        rows.append(flatten(chunk, label=ep['etype']))
        # Flatten the chunk and add to the rows list with the episode type as label

# ---------- 6.  Negative windows ----------
pos_starts = episodes['start_core']
# Get the start times of all positive episodes
all_idx    = feat.index[feat.index >= (feat.index[0] + offset)]       # ignore first 2 h
# Get all possible indices after the initial context window
neg_idx    = all_idx.difference(pos_starts)
# Exclude indices that are positive episode starts

# drop anything within ±CTX_MIN of an onset
for s in pos_starts:
    # For each positive episode start
    neg_idx = neg_idx[(neg_idx < s - offset) | (neg_idx > s + offset)]
    # Remove indices within the context window before or after any onset

neg_picks = rng.choice(
    # Randomly select negative window start indices
    neg_idx,
    size=min(len(rows)*2, len(neg_idx)),   # 2 neg per pos
    # Pick up to twice as many negatives as positives, or as many as available
    replace=False
)

for t0 in neg_picks:
    start_idx = feat.index.asof(t0 - offset)
    end_idx = feat.index.asof(t0)
    if pd.isna(start_idx) or pd.isna(end_idx):
        continue  # skip if no valid index
    chunk = feat.loc[start_idx:end_idx].tail(WIN_STEPS)
    if len(chunk) == WIN_STEPS:
        rows.append(flatten(chunk, label=0))
        # Flatten the chunk and add to the rows list with label 0 (negative)

# ---------- 7.  Save ----------
out = pd.DataFrame(rows)
# Convert the list of window dictionaries to a DataFrame
out.to_csv("compression_low_ml_windows.csv", index=False)
# Save the DataFrame to a CSV file

print(f"Saved compression_low_ml_windows.csv – "
      f"{(out['label']>0).sum()} positives / {(out['label']==0).sum()} negatives")
# Print a summary of the number of positive and negative windows saved
