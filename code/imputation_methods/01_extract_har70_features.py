# Copyright (c) 2025 Gabriel-Vasilica Sasu
# All rights reserved.
#
# 01_extract_har70_features.py
# ---------------------------------------------------------------------------
# STEP 1 of the HAR70+ external-validation pipeline.
#
# Converts the raw HAR70+ accelerometer recordings (50 Hz) into a windowed
# feature dataset that has the same structure as the wearable dataset used in
# the paper: one row = one 3-second window, described by 19 numeric features,
# plus the majority Activity label.
#
# Why windows (not raw samples):
#   Raw 50 Hz samples are near-duplicates of their neighbours, which makes
#   nearest-neighbour imputation trivially "perfect" and inflates results.
#   Windowed features (per-channel mean/std, magnitudes, cross-axis
#   correlations) give a realistic multivariate problem and match how HAR70+
#   is used in the literature.
#
# INPUT:  ./har70plus/*.csv   (the 18 per-participant files from the HAR70+ zip)
# OUTPUT: ./data/har70_full.csv
# ---------------------------------------------------------------------------

import numpy as np, pandas as pd, glob, os

RAW_DIR = "har70plus"          # folder with 501.csv ... 518.csv
OUT_DIR = "data"
OUT     = os.path.join(OUT_DIR, "har70_full.csv")
WIN     = 150                  # 3 s @ 50 Hz
STEP    = 150                  # non-overlapping windows
MIN_HOMOGENEITY = 0.80         # keep a window only if >=80% of samples share the label

CH = ["back_x","back_y","back_z","thigh_x","thigh_y","thigh_z"]

# HAR70+ label codes -> grouped activity names.
# 1 walking, 3 shuffling, 4 stairs-up, 5 stairs-down, 6 standing, 7 sitting, 8 lying.
# Rare stair classes are merged; the resulting "stairs" group is dropped later if too small.
LABEL_MAP = {1:"walking", 3:"walking", 4:"stairs", 5:"stairs",
             6:"standing", 7:"sitting", 8:"lying"}
DROP_IF_RARE = "stairs"        # dropped if it ends up < 1% of windows

def window_features(w):
    arr = w[CH].to_numpy(float)
    f = {}
    for i, c in enumerate(CH):
        x = arr[:, i]
        f[f"{c}_mean"] = x.mean()
        f[f"{c}_std"]  = x.std()
    back_mag  = np.sqrt((arr[:, 0:3] ** 2).sum(axis=1))
    thigh_mag = np.sqrt((arr[:, 3:6] ** 2).sum(axis=1))
    f["back_mag_mean"]  = back_mag.mean();  f["back_mag_std"]  = back_mag.std()
    f["thigh_mag_mean"] = thigh_mag.mean(); f["thigh_mag_std"] = thigh_mag.std()
    def corr(a, b):
        return float(np.corrcoef(a, b)[0, 1]) if np.std(a) * np.std(b) > 1e-8 else 0.0
    f["corr_back_xy"]    = corr(arr[:, 0], arr[:, 1])
    f["corr_thigh_xy"]   = corr(arr[:, 3], arr[:, 4])
    f["corr_back_thigh"] = corr(back_mag, thigh_mag)
    return f

def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    files = sorted(glob.glob(os.path.join(RAW_DIR, "*.csv")))
    if not files:
        raise SystemExit(f"No CSV files found in ./{RAW_DIR}/ . "
                         f"Unzip the HAR70+ archive so that {RAW_DIR}/501.csv ... 518.csv exist.")
    rows = []
    for path in files:
        pid = int(os.path.basename(path).split(".")[0])
        df = pd.read_csv(path)
        n = len(df)
        for start in range(0, n - WIN + 1, STEP):
            w = df.iloc[start:start + WIN]
            vals, counts = np.unique(w["label"].values, return_counts=True)
            maj = vals[counts.argmax()]
            if counts.max() < MIN_HOMOGENEITY * WIN or maj not in LABEL_MAP:
                continue
            feat = window_features(w)
            feat["Activity"] = LABEL_MAP[maj]
            feat["participant"] = pid
            rows.append(feat)
        print(f"  participant {pid}: cumulative windows = {len(rows)}", flush=True)

    out = pd.DataFrame(rows)
    # drop the rare merged class if it is negligible
    counts = out["Activity"].value_counts(normalize=True)
    if DROP_IF_RARE in counts.index and counts[DROP_IF_RARE] < 0.01:
        out = out[out["Activity"] != DROP_IF_RARE].reset_index(drop=True)
        print(f"Dropped rare class '{DROP_IF_RARE}'.")

    feat_cols = [c for c in out.columns if c not in ("Activity", "participant")]
    out.to_csv(OUT, index=False)
    print(f"\nSaved {OUT}")
    print(f"windows: {len(out)} | numeric features: {len(feat_cols)} | "
          f"participants: {out['participant'].nunique()}")
    print("activity distribution:", out["Activity"].value_counts().to_dict())

if __name__ == "__main__":
    main()
