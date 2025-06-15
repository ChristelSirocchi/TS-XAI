import numpy as np
import pandas as pd
import argparse
import gc
from tqdm import tqdm
from utilities import *

# ----------------------- Parse command-line arguments -----------------------
parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str, default="physionet")
parser.add_argument("--dept", type=str, default="ALL")
parser.add_argument("--task", type=str, default="HOSPITAL_EXPIRE_FLAG")
parser.add_argument("--distance", type=str, default="euclidean")
parser.add_argument("--computed", type=str, default="mean")  # imputed
parser.add_argument("--MAX", type=int, default=5)
parser.add_argument("--interval", type=int, default=1)
parser.add_argument("--percentile", type=float, default=0.05)
args = parser.parse_args()

# ----------------------- Load parameters -----------------------
dataset = args.dataset
distance = args.distance
computed = args.computed
DEPT = args.dept
TASK = args.task
MAX = args.MAX
interval = args.interval
percentile = args.percentile
exp_name = DEPT + "_" + TASK
suffix = f"_p{int(percentile*100)}" if percentile != 0.05 else ""
interval_suffix = f"_{interval}" if interval != 1 else ""

print(f"[Checkpoint] Dataset: {dataset}, Experiment: {exp_name}, Distance: {distance}, Task: {TASK}, Percentile: {percentile}")

# ----------------------- Load admission and task data -----------------------
adms = pd.read_csv(f"{dataset}/sub_adm_target.csv", index_col=0)
adm_ids = adms["HADM_ID"].unique()
task_dict = adms[["HADM_ID", TASK]].set_index("HADM_ID")[TASK].to_dict()

# ----------------------- Load pairwise distances -----------------------
dist_path = f"{dataset}/{exp_name}/{distance}/{computed}_distances{interval_suffix}.npz"

print(f"[Checkpoint] Loading distances from {dist_path}")
dists = np.load(dist_path)["mean"].astype(np.float32)

# ----------------------- Filter closest pairs by percentile -----------------------
low_q = np.quantile(dists, percentile)
n = len(adm_ids)
i, j = np.triu_indices(n, k=1)
mask = (dists < low_q)

filtered_pairs_df = pd.DataFrame({
    "adm1": adm_ids[i[mask]],
    "adm2": adm_ids[j[mask]],
    "distance": dists[mask]
})
filtered_pairs_df["tg_adm1"] = filtered_pairs_df["adm1"].map(task_dict)
filtered_pairs_df["tg_adm2"] = filtered_pairs_df["adm2"].map(task_dict)

# ----------------------- Select positive and similar-distance same-target pairs -----------------------
pos_pairs = filtered_pairs_df[filtered_pairs_df["tg_adm1"] != filtered_pairs_df["tg_adm2"]]
same_target_pairs = filtered_pairs_df[filtered_pairs_df["tg_adm1"] == filtered_pairs_df["tg_adm2"]]

sel_pos_pairs = select_pairs(pos_pairs, MAX)
print(f"[Checkpoint] Selected {len(sel_pos_pairs)} positive pairs")

# ----------------------- Match control pairs -----------------------
selected_rows = []
for _, row in tqdm(sel_pos_pairs.iterrows(), total=len(sel_pos_pairs), desc="Processing pairs"):
    a1, a2 = row["adm1"], row["adm2"]
    d = row["distance"]
    epsilon = d * 0.1
    t1, t2 = row["tg_adm1"], row["tg_adm2"]

    same1 = same_target_pairs[
        (((same_target_pairs["adm1"] == a1) & (same_target_pairs["tg_adm2"] == t1)) |
         ((same_target_pairs["adm2"] == a1) & (same_target_pairs["tg_adm1"] == t1))) &
        (np.abs(same_target_pairs["distance"] - d) <= epsilon)
    ]

    same2 = same_target_pairs[
        (((same_target_pairs["adm1"] == a2) & (same_target_pairs["tg_adm2"] == t2)) |
         ((same_target_pairs["adm2"] == a2) & (same_target_pairs["tg_adm1"] == t2))) &
        (np.abs(same_target_pairs["distance"] - d) <= epsilon)
    ]

    if (not same1.empty) and (not same2.empty):
        selected_rows.append(row)
        idx1 = (same1["distance"] - d).abs().idxmin()
        selected_rows.append(same1.loc[idx1])
        same_target_pairs = same_target_pairs.drop(idx1)
        idx2 = (same2["distance"] - d).abs().idxmin()
        selected_rows.append(same2.loc[idx2])
        same_target_pairs = same_target_pairs.drop(idx2)

sel_pairs_df = pd.DataFrame(selected_rows)
sel_pairs_df["target"] = [1, 0, 0] * int(len(sel_pairs_df) / 3)

# ----------------------- Save selected pairs -----------------------
pair_path = f"{dataset}/{exp_name}/{distance}/sel_pairs_{computed}_{MAX}_{interval}{suffix}.csv"
sel_pairs_df.to_csv(pair_path, index=False)
print(f"[Checkpoint] Saved selected pairs")

# ----------------------- Load features and compute differences -----------------------
del filtered_pairs_df, pos_pairs, same_target_pairs
features_df = pd.read_csv(f"{dataset}/{exp_name}/ts_features.csv", index_col=0, header=[0, 1])

feature_diff_list = []
for _, row in sel_pairs_df.iterrows():
    f1 = features_df.loc[row["adm1"]]
    f2 = features_df.loc[row["adm2"]]
    diff = np.abs(f1 - f2)
    feature_diff_list.append(diff)

feature_diff_df = pd.DataFrame(feature_diff_list, columns=features_df.columns)

# ----------------------- Train random forest -----------------------
X = feature_diff_df.values
y = sel_pairs_df["target"].values

best_params = {
    'n_estimators': 200,
    'max_depth': None,
    'min_samples_split': 10,
    'class_weight': None
}

param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [10, None],
    'min_samples_split': [5, 10],
    'class_weight': [
        None,
        {0: 0.5, 1: 1},
        {0: 0.5, 1: 2}
    ]
}

print(f"[Checkpoint] Running Random Forest cross-validation")
results_rf, feat_imp_rf = run_rf_cv(X, y, False, 42, param_grid, best_params, n_splits=3)

# ----------------------- Save results -----------------------
results_path = f"{dataset}/{exp_name}/{distance}/results_rf_{MAX}_{interval}{suffix}.csv"
imp_path = f"{dataset}/{exp_name}/{distance}/feat_imp_rf_{computed}_{MAX}_{interval}{suffix}.csv"

pd.DataFrame(results_rf).to_csv(results_path, index=False)
pd.DataFrame(feat_imp_rf, columns=feature_diff_df.columns).T.reset_index().to_csv(imp_path, index=False)

print(f"[Checkpoint] Saved RF results and feature importances")

