import numpy as np
import pandas as pd
import argparse
import gc
from tqdm import tqdm
from utilities import *
from sklearn.preprocessing import QuantileTransformer
import warnings
warnings.filterwarnings("ignore")

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
parser.add_argument("--pairs", type=str, default="matched")
parser.add_argument("--features", type=str, default="measured")
args = parser.parse_args()

# ----------------------- Load parameters -----------------------
dataset = args.dataset
distance = args.distance
computed = args.computed
DEPT = args.dept
TASK = args.task
MAX = args.MAX
interval = args.interval
pairs = args.pairs
features = args.features
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
dist_path = f"{dataset}/{exp_name}/{distance}/{computed}_distances{interval_suffix}_out.npz"

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
pos_pairs = filtered_pairs_df[filtered_pairs_df["tg_adm1"] != filtered_pairs_df["tg_adm2"]].sort_values("distance")
same_target_pairs = filtered_pairs_df[filtered_pairs_df["tg_adm1"] == filtered_pairs_df["tg_adm2"]].sort_values("distance")

min_d = pos_pairs["distance"].min()
max_d = pos_pairs["distance"].max()
same_target_pairs = same_target_pairs[(same_target_pairs["distance"] >= min_d) & (same_target_pairs["distance"] <= max_d)]


if pairs == "matched":
    sel_pos_pairs = select_pairs_fast(pos_pairs, MAX)
    sel_pairs_df = select_matched_pairs_fast(sel_pos_pairs, same_target_pairs)

else:
    pos_pairs = select_pairs_fast(pos_pairs, MAX)
    pos_pairs["target"] = 1

    neg_pairs = select_pairs_fast(same_target_pairs, MAX)
    neg_pairs["target"] = 0

    sel_pairs_df = pd.concat([pos_pairs,neg_pairs])

pair_path = f"{dataset}/{exp_name}/{distance}/sel_pairs_{computed}_{MAX}_{pairs}_{features}{interval_suffix}{suffix}_out.csv"
sel_pairs_df.to_csv(pair_path, index=False)
print(f"[Checkpoint] Saved selected pairs")

del filtered_pairs_df, pos_pairs, same_target_pairs

# ----------------------- Load features and compute differences -----------------------

features_df = pd.read_csv(f"{dataset}/{exp_name}/ts_features_out.csv", index_col=0, header=[0, 1])

if features == "quantile":
    qt = QuantileTransformer(n_quantiles=100, output_distribution='normal')
    features_df = pd.DataFrame(
        qt.fit_transform(features_df.values),
        index=features_df.index,
        columns=features_df.columns
    )
    print(f"[Checkpoint] Quantile transformation")

feature_diff_list = []
for _, row in sel_pairs_df.iterrows():
    f1 = features_df.loc[row["adm1"]]
    f2 = features_df.loc[row["adm2"]]
    diff = np.abs(f1 - f2)
    feature_diff_list.append(diff)
feature_diff_df = pd.DataFrame(feature_diff_list, columns=features_df.columns)

# ----------------------- Train random forest -----------------------
sel_pairs_df["tgcc"] = sel_pairs_df["tg_adm1"] + sel_pairs_df["tg_adm2"]

X = feature_diff_df.values
y = sel_pairs_df["target"].values
z = sel_pairs_df["tgcc"].values

best_params = {
    'n_estimators': 100,
    'max_depth': None,
    'min_samples_split': 5,
    'class_weight': {0: 1, 1: 5}
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
results_rf, feat_imp_rf = run_rf_cv(X, y, z, False, 42, param_grid, best_params, n_splits=3)

# ----------------------- Save results -----------------------
results_path = f"{dataset}/{exp_name}/{distance}/results_rf_{computed}_{MAX}_{pairs}_{features}{interval_suffix}{suffix}_out.csv"
imp_path = f"{dataset}/{exp_name}/{distance}/feat_imp_rf_{computed}_{MAX}_{pairs}_{features}{interval_suffix}{suffix}_out.csv"

pd.DataFrame(results_rf).to_csv(results_path, index=False)

imps = pd.DataFrame(feat_imp_rf, columns=feature_diff_df.columns).T.mean(axis=1).reset_index().sort_values(by=0, ascending=False)
imps.columns = ["time","label","imp"]
feature_diff_df["target"] = y
mean_comp = feature_diff_df.groupby("target").mean().T.reset_index()
mean_comp["diff"] = mean_comp[1]-mean_comp[0]
mean_comp.columns = ["time","label","0", "1", "diff"]
imps = imps.merge(mean_comp)
imps.to_csv(imp_path, index=False)

print(f"[Checkpoint] Saved RF results and feature importances")

