import pandas as pd, numpy as np, joblib
from compression_onset import make_features

df = pd.read_csv(r"AI in Health/sample1.csv")
df["glucose"] = pd.to_numeric(df["glucose"], errors="coerce")  # Ensure glucose is float
g = df["glucose"].to_numpy(float)

bundle = joblib.load("AI in Health/model_bundle.joblib")
X = make_features(g, bundle["win_var"], bundle["delta_short"], bundle["delta_pct"])
X = bundle["scaler"].transform(X)
# Remove rows with NaN values in X and align df
mask = ~np.isnan(X).any(axis=1)
X = X[mask]
df = df.loc[mask].reset_index(drop=True)
P = bundle["model"].predict_proba(X)  # cols: 0 none, 1 comp, 2 reg

pred = np.zeros(len(df), dtype=int)
pred[(P[:,1] > bundle["T_comp"])] = 1
pred[(P[:,2] > bundle["T_reg"]) & (P[:,1] <= bundle["T_comp"])] = 2

df["p_comp"] = P[:,1]
df["p_reg"]  = P[:,2]
df["pred"]   = pred
df.to_csv("preds.csv", index=False)
print("Wrote preds.csv")
