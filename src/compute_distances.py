#!/usr/bin/env python3
import argparse
import pandas as pd
import numpy as np
import os
from scipy.spatial.distance import pdist
from sklearn.metrics import silhouette_score

# Parse command-line arguments
parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str, default="physionet")
parser.add_argument("--dept", type=str, default="ALL")
parser.add_argument("--task", type=str, default="HOSPITAL_EXPIRE_FLAG")
parser.add_argument("--distance", type=str, default="euclidean")
parser.add_argument("--interval", type=int, default=1)
args = parser.parse_args()

dataset = args.dataset
distance = args.distance
DEPT = args.dept
TASK = args.task
interval = args.interval

# Construct experiment name and suffix
exp_name = f"{DEPT}_{TASK}"
interval_suffix = f"_{interval}" if interval != 1 else ""

print(f"Dataset: {dataset}, Experiment: {exp_name}, Distance: {distance}, Task: {TASK}")

# Load admission-level labels
adms = pd.read_csv(f"{dataset}/adm_target.csv")
if DEPT != "ALL":
    adms = adms[adms["ADMISSION_TYPE"] == DEPT]

# Load discrete time series
df = pd.read_csv(f"{dataset}/{exp_name}/discrete_ts{interval_suffix}.csv", index_col=[0, 1])

print(f"[Checkpoint] Data imported.")

# --- Compute distances on available data --- #

# Prepare for pairwise distance computation
labs = df.index.get_level_values(1).unique()
adm_ids_all = df.index.get_level_values(0).unique().to_numpy()
n = len(adm_ids_all)

# Get indices of upper triangle for pairwise distances
i, j = np.triu_indices(n, k=1)
shared_true_counts = np.zeros_like(i, dtype=np.int32)
dist_sums = np.zeros_like(i, dtype=np.float32)

# Unstack time series and fill missing values with 0
df_unstacked = df.unstack(level=-1).fillna(0).astype(np.float32)
df_unstacked.columns = df_unstacked.columns.swaplevel(0, 1)
df_unstacked = df_unstacked.sort_index(axis=1)

# Create mask of missing values for each admission
df_bool = df["0"].unstack(level=-1).isna().values.astype(bool)
shared_true_counts = np.sum(~df_bool[i] & ~df_bool[j], axis=1)

# Compute distances for each lab test and accumulate
for lab in labs:
    print(f"Processing lab: {lab}")
    try:
        X = df_unstacked[lab].values  # shape: (n_adm, timepoints)
        dist = pdist(X, metric=distance)  # shape: (n_pairs,)
        dist_sums += dist
    except KeyError:
        continue  # Lab not found in data, skip

# Compute average distance per pair (only where overlap exists)
with np.errstate(divide='ignore', invalid='ignore'):
    avg_dist = np.divide(dist_sums, shared_true_counts, out=np.zeros_like(dist_sums), where=shared_true_counts > 0)
    avg_dist[shared_true_counts == 0] = np.nan

# Ensure output directory exists
os.makedirs(f"{dataset}/{exp_name}/{distance}", exist_ok=True)

# Save averaged distances
np.savez_compressed(f"{dataset}/{exp_name}/{distance}/mean_distances{interval_suffix}.npz", mean=avg_dist)

print(f"[Checkpoint] Mean distance over available labs saved.")

# --- Compute distances on imputed data --- #

# Load imputed time series (wide format: admission x lab:time)
df_imputed = pd.read_csv(f"{dataset}/{exp_name}/ts_imputed{interval_suffix}.csv", index_col=0, header=[0, 1])

# Compute full pairwise distances on imputed time series
dists = pdist(df_imputed.values.astype(np.float32), metric=distance)
np.savez_compressed(f"{dataset}/{exp_name}/{distance}/imputed_distances{interval_suffix}.npz", mean=dists)

print(f"[Checkpoint] Distance over all imputed labs saved.")
