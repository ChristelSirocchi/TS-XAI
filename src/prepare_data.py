import os
import numpy as np
import pandas as pd
import argparse
from tqdm import tqdm
from sklearn.impute import KNNImputer
from utilities import *

# ----------------- Parse command-line arguments -----------------
parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str, default="physionet")
parser.add_argument("--dept", type=str, default="ALL")
parser.add_argument("--task", type=str, default="HOSPITAL_EXPIRE_FLAG")
parser.add_argument("--interval", type=int, default=1)
args = parser.parse_args()

dataset = args.dataset
DEPT = args.dept
TASK = args.task
interval = args.interval
exp_name = f"{DEPT}_{TASK}"

interval_suffix = f"_{interval}" if interval != 1 else ""

print(f"[Checkpoint] Dataset: {dataset}, Experiment: {exp_name}, Task: {TASK}")

output_dir = os.path.join(dataset, exp_name)
os.makedirs(output_dir, exist_ok=True)

# ----------------- Read input data -----------------
print("[Checkpoint] Reading input CSVs...")
all_labs = pd.read_csv(f"{dataset}/all_labs.csv", index_col=0)
adms = pd.read_csv(f"{dataset}/adm_target.csv", index_col=0)

# ----------------- Preprocess admissions and labs -----------------
all_labs["HADM_ID"] = all_labs["HADM_ID"].astype(int)
adms["HADM_ID"] = adms["HADM_ID"].astype(int)

if DEPT == "ALL":
    df_adm1 = adms
else:
    df_adm1 = adms[adms["ADMISSION_TYPE"] == DEPT]

df_lab1 = all_labs[all_labs["HADM_ID"].isin(df_adm1["HADM_ID"])]

nlabs = all_labs["label"].nunique()
gg = df_lab1.groupby(["HADM_ID", "label"]).size().reset_index().groupby("HADM_ID").size().sort_values(ascending=False)
freq_adms = gg[gg >= nlabs * 0.50].index.astype(int)

df_adm1 = df_adm1[df_adm1["HADM_ID"].isin(freq_adms)]
df_lab1 = all_labs[all_labs["HADM_ID"].isin(df_adm1["HADM_ID"])]

df_adm1 = df_adm1.sort_values("HADM_ID").reset_index(drop=True)

df_adm1.to_csv(f"{dataset}/sub_adm_target.csv")
df_lab1.to_csv(f"{dataset}/sub_adm_lab.csv")

print("[Checkpoint] Filtered admissions and labs saved.")

# ----------------- Time Series Discretisation -----------------
sub_df_lab = df_lab1.copy()
sub_df_lab["HADM_ID"] = sub_df_lab["HADM_ID"].astype(int)
sub_df_lab["hour"] = sub_df_lab["hour"].astype(int)
sub_df_lab.dropna(subset=["HADM_ID", "label", "hour", "VALUENUM"], inplace=True)

# Compute stats per lab label for normalisation
label_stats = sub_df_lab.groupby("label")["VALUENUM"].agg(["mean", "std"]).rename(columns={"mean": "label_mean", "std": "label_std"})
sub_df_lab = sub_df_lab.merge(label_stats, on="label")

# Remove constant-valued lab labels
sub_df_lab = sub_df_lab[sub_df_lab["label_std"] > 0]

# Discretise by interval
sub_df_lab["interval"] = (sub_df_lab["hour"] // interval) * interval

print("[Checkpoint] Discretising time series...")

# Pivot time series
df1 = (
    sub_df_lab
    .groupby(["HADM_ID", "label", "label_mean", "label_std", "interval"])["VALUENUM"]
    .mean()
    .unstack(level=-1)
    .interpolate(method='linear', axis=1)
    .ffill(axis=1)
    .bfill(axis=1)
    .reset_index()
)

# Normalise values
time_bins = list(range(0, 49, interval))
df1[time_bins] = df1[time_bins].subtract(df1["label_mean"], axis=0).divide(df1["label_std"], axis=0)

# Store and index
df1 = df1.drop(columns=["label_mean", "label_std"]).set_index(["HADM_ID", "label"])
df1.to_csv(f"{dataset}/{exp_name}/discrete_ts{interval_suffix}.csv")

print("[Checkpoint] Discrete normalised time series saved.")

# ----------------- KNN Imputation -----------------
print("[Checkpoint] Performing KNN imputation...")

df_unstacked = df1.unstack(level=-1)
df_unstacked.columns = df_unstacked.columns.swaplevel(0, 1)
df_unstacked = df_unstacked.sort_index(axis=1)

imputer = KNNImputer(n_neighbors=5)
df_imputed = pd.DataFrame(
    imputer.fit_transform(df_unstacked),
    index=df_unstacked.index,
    columns=df_unstacked.columns
)

df_imputed.to_csv(f"{dataset}/{exp_name}/ts_imputed{interval_suffix}.csv")

# ----------------- Feature computation -----------------

print("[Checkpoint] KNN-imputed time series saved.")

features_df = compute_ts_metrics_window(sub_df_lab)
features_df.set_index(["HADM_ID", "label"], inplace=True)
features_df = features_df.unstack(level=-1)

features_df.to_csv(f"{dataset}/{exp_name}/ts_features.csv")

print("[Checkpoint] Temporal features computed.")

