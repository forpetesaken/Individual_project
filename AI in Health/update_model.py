import pandas as pd, numpy as np, joblib, os
from sklearn.linear_model import SGDClassifier
from sklearn.utils.class_weight import compute_class_weight
from compression_onset import make_features, pick_threshold  # already in your file

CSV_MAIN   = "sample1.csv"        # your full, original labeled data
CSV_FEED   = "feedback.csv"       # your corrections
BUNDLE_IN  = "model_bundle.joblib"
BUNDLE_OUT = "model_bundle.joblib"  # overwrite same file

# ---------- 1) Load feedback ----------
fb = pd.read_csv(CSV_FEED)
if "timestamp" in fb.columns:
    fb = fb.sort_values("timestamp").reset_index(drop=True)
    key_cols = ["timestamp","glucose"]
else:
    # fall back to row index if no timestamp
    fb["idx"] = fb.index
    key_cols = ["idx","glucose"]

g_fb  = fb["glucose"].to_numpy(float)
y_fb  = fb["true_label"].astype(int).to_numpy()

# ---------- 2) Load existing bundle ----------
bundle = joblib.load(BUNDLE_IN)
scaler = bundle["scaler"]
model  = bundle["model"]

# ---------- 3) Ensure model supports partial_fit ----------
if not isinstance(model, SGDClassifier):
    print("Swapping to SGDClassifier (one-time).")
    full = pd.read_csv(CSV_MAIN)
    if "timestamp" in full.columns:
        full = full.sort_values("timestamp").reset_index(drop=True)
    else:
        full["timestamp"] = pd.date_range("2000-01-01", periods=len(full), freq="1min")

    g_full = full["glucose"].to_numpy(float)
    y_full = full["label"].astype(int).to_numpy()

    # rebuild features & (re)fit scaler
    X_full = make_features(g_full, bundle["win_var"], bundle["delta_short"], bundle["delta_pct"])
    X_full = scaler.fit_transform(X_full)

    
    classes = np.array([0,1,2])
    cw = compute_class_weight(class_weight="balanced", classes=classes, y=y_full)
    cw_map = dict(zip(classes, cw))
    sw_full = np.vectorize(cw_map.get)(y_full)

    model = SGDClassifier(loss="log_loss", random_state=0)
    model.partial_fit(X_full, y_full, classes=classes, sample_weight=sw_full)


# ---------- 4) Partial fit on feedback ----------
X_fb = make_features(g_fb, bundle["win_var"], bundle["delta_short"], bundle["delta_pct"])
X_fb = scaler.transform(X_fb)
sw_fb = np.vectorize(cw_map.get)(y_fb)  # reuse cw_map from above
model.partial_fit(X_fb, y_fb, sample_weight=sw_fb)


# ---------- 5) (Optional) re-pick thresholds ----------
# Quick: use last 20% of main data
full = pd.read_csv(CSV_MAIN)
if "timestamp" in full.columns:
    full = full.sort_values("timestamp").reset_index(drop=True)
ts = pd.to_datetime(full.get("timestamp", pd.RangeIndex(len(full))))
g  = full["glucose"].to_numpy(float)
y  = full["label"].astype(int).to_numpy()

X  = scaler.transform(make_features(g, bundle["win_var"], bundle["delta_short"], bundle["delta_pct"]))
P  = model.predict_proba(X)

split = int(len(X)*0.8)
y_val  = y[split:]
ts_val = ts[split:]
p_comp = P[split:,1]
p_reg  = P[split:,2]

best_comp = pick_threshold((y_val==1).astype(int), p_comp, ts_val, max_false_per_day=bundle.get("fa_day_comp",0.5))
best_reg  = pick_threshold((y_val==2).astype(int), p_reg , ts_val, max_false_per_day=bundle.get("fa_day_reg",1.0))

bundle["T_comp"] = best_comp["T"]
bundle["T_reg"]  = best_reg["T"]
bundle["model"]  = model

joblib.dump(bundle, BUNDLE_OUT)
print("Updated model saved.")
print(f"New T_comp={bundle['T_comp']:.3f}, T_reg={bundle['T_reg']:.3f}")
