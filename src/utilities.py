import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from imblearn.over_sampling import RandomOverSampler 
from imblearn.under_sampling import RandomUnderSampler
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score, balanced_accuracy_score, precision_recall_curve, auc
from scipy.stats import linregress
from scipy.signal import find_peaks
from sklearn.metrics.pairwise import cosine_similarity
from scipy.spatial.distance import pdist, squareform
from collections import defaultdict
from tqdm import tqdm

# function to train random forest
def run_dt_cv(X, y, GRID=True, SEED=42, param_grid=None, best_params=None, n_splits=5):
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=SEED)
    results = []
    feat_imp = []

    for i, (train_idx, test_idx) in enumerate(skf.split(X, y)):
        print(f"Fold {i + 1}")

        x_train, x_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        # Oversample minority class
        oversample = RandomOverSampler(sampling_strategy='minority', random_state=SEED)
        x_train, y_train = oversample.fit_resample(x_train, y_train)

        if GRID:
            dt = GridSearchCV(
                DecisionTreeClassifier(random_state=SEED),
                param_grid,
                scoring='roc_auc',
                cv=3,
                n_jobs=-1
            )
            dt.fit(x_train, y_train)
            best_dt = dt.best_estimator_
            print("Best parameters:", dt.best_params_)
        else:
            best_dt = DecisionTreeClassifier(**best_params, random_state=SEED)
            best_dt.fit(x_train, y_train)

        pred = best_dt.predict(x_test)
        prob = best_dt.predict_proba(x_test)[:, 1]

        metrics = {
            'fold': i + 1,
            'roc_auc': roc_auc_score(y_test, prob),
            'pr_auc': auc(*precision_recall_curve(y_test, prob)[1::-1]),
            'recall': recall_score(y_test, pred),
            'precision': precision_score(y_test, pred),
            'balanced_accuracy': balanced_accuracy_score(y_test, pred),
            'f1': f1_score(y_test, pred)
        }
        results.append(metrics)
        print(metrics)
        feat_imp.append(best_dt.feature_importances_)

    return results, feat_imp


# function to train random forest
def run_rf_cv(X, y, z=None, GRID=True, SEED=42, param_grid=None, best_params=None, n_splits=5):
    if z is None:
        z = y
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=SEED)
    results = []
    feat_imp = []

    for i, (train_idx, test_idx) in enumerate(skf.split(X, z)):
        print(f"Fold {i + 1}")

        x_train, x_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        # Oversample minority class
        #oversample = RandomOverSampler(sampling_strategy='minority', random_state=SEED)
        #x_train, y_train = oversample.fit_resample(x_train, y_train)
        
        undersample = RandomUnderSampler(sampling_strategy='majority', random_state=SEED)
        x_train, y_train = undersample.fit_resample(x_train, y_train)

        if GRID:
            dt = GridSearchCV(
                RandomForestClassifier(random_state=SEED),
                param_grid,
                scoring='roc_auc',
                cv=3,
                n_jobs=-1
            )
            dt.fit(x_train, y_train)
            best_dt = dt.best_estimator_
            print("Best parameters:", dt.best_params_)
        else:
            best_dt = RandomForestClassifier(**best_params, random_state=SEED)
            best_dt.fit(x_train, y_train)

        pred = best_dt.predict(x_test)
        prob = best_dt.predict_proba(x_test)[:, 1]

        metrics = {
            'fold': i + 1,
            'roc_auc': roc_auc_score(y_test, prob),
            'pr_auc': auc(*precision_recall_curve(y_test, prob)[1::-1]),
            'recall': recall_score(y_test, pred),
            'precision': precision_score(y_test, pred),
            'balanced_accuracy': balanced_accuracy_score(y_test, pred),
            'f1': f1_score(y_test, pred)
        }
        results.append(metrics)
        print(metrics)
        feat_imp.append(best_dt.feature_importances_)

    return results, feat_imp

def select_pairs(df_sorted, N):
    all_adms = list(set(df_sorted["adm1"]) | set(df_sorted["adm2"]))
    admission_counts = {adm: 0 for adm in all_adms}
    
    selected_rows = []
    
    for _, row in df_sorted.iterrows():
        a1, a2 = row['adm1'], row['adm2']
        if admission_counts[a1] < N and admission_counts[a2] < N:
            selected_rows.append(row)
            admission_counts[a1] += 1
            admission_counts[a2] += 1
            if all(count >= N for count in admission_counts.values()):
                break
                
    return pd.DataFrame(selected_rows)

def select_pairs_fast(df_sorted, N):
    adms = pd.Index(df_sorted['adm1'].tolist() + df_sorted['adm2'].tolist()).unique()
    adm_to_idx = {adm: i for i, adm in enumerate(adms)}
    counts = np.zeros(len(adms), dtype=int)

    selected_indices = []

    adm1_idx = df_sorted["adm1"].map(adm_to_idx).to_numpy()
    adm2_idx = df_sorted["adm2"].map(adm_to_idx).to_numpy()

    for i in tqdm(range(len(df_sorted)), desc="Selecting pairs"):
        i1, i2 = adm1_idx[i], adm2_idx[i]
        if counts[i1] < N and counts[i2] < N:
            selected_indices.append(i)
            counts[i1] += 1
            counts[i2] += 1
            if np.all(counts >= N):
                break

    return df_sorted.iloc[selected_indices].copy()

def select_matched_pairs_fast(sel_pos_pairs, same_target_pairs):
    selected_rows = []

    # Make a copy of same_target_pairs and add a flag to track used rows
    same_target_pairs = same_target_pairs.copy()
    same_target_pairs["used"] = False

    # Convert relevant columns to numpy arrays for faster access
    same_adm1 = same_target_pairs["adm1"].to_numpy()
    same_adm2 = same_target_pairs["adm2"].to_numpy()
    same_tg1 = same_target_pairs["tg_adm1"].to_numpy()
    same_tg2 = same_target_pairs["tg_adm2"].to_numpy()
    same_dist = same_target_pairs["distance"].to_numpy()
    used = same_target_pairs["used"].to_numpy()

    for i, row in tqdm(sel_pos_pairs.iterrows(), total=len(sel_pos_pairs), desc="Matching pairs"):
        a1, a2 = row["adm1"], row["adm2"]
        t1, t2 = row["tg_adm1"], row["tg_adm2"]
        d = row["distance"]
        epsilon = d * 0.1

        # Find indices of valid rows for a1
        mask1 = (
            (~used) &
            (((same_adm1 == a1) & (same_tg2 == t1)) | ((same_adm2 == a1) & (same_tg1 == t1))) &
            (np.abs(same_dist - d) <= epsilon)
        )

        # Find indices of valid rows for a2
        mask2 = (
            (~used) &
            (((same_adm1 == a2) & (same_tg2 == t2)) | ((same_adm2 == a2) & (same_tg1 == t2))) &
            (np.abs(same_dist - d) <= epsilon)
        )

        idxs1 = np.where(mask1)[0]
        idxs2 = np.where(mask2)[0]

        if idxs1.size > 0 and idxs2.size > 0:
            # Keep original row (different targets)
            selected_rows.append(row)

            # Closest matching row for a1
            best1 = idxs1[np.argmin(np.abs(same_dist[idxs1] - d))]
            selected_rows.append(same_target_pairs.iloc[best1])
            used[best1] = True

            # Closest matching row for a2
            best2 = idxs2[np.argmin(np.abs(same_dist[idxs2] - d))]
            selected_rows.append(same_target_pairs.iloc[best2])
            used[best2] = True

    sel_pairs_df = pd.DataFrame(selected_rows)
    sel_pairs_df["target"] = [1, 0, 0] * (len(sel_pairs_df) // 3)
    return sel_pairs_df

def compute_ts_stats(times, values):
    # basic stats
    minimum = np.min(values)
    maximum = np.max(values)
    mean = np.mean(values)
    std = np.std(values)
    value_range = maximum - minimum

    # trend (slope from linear regression)
    try:
        if len(values) > 1:
            slope, intercept, _, _, _ = linregress(times, values)
        else:
            slope, intercept = np.nan, np.nan
    except ValueError:
        slope, intercept = np.nan, np.nan

    # pattern features
    #peaks, _ = find_peaks(values)
    #num_peaks = len(peaks)

    return  {
            "mean": mean,
            "std": std,
            "min": minimum,
            "max": maximum,
            "range": value_range,
            "slope": slope,
            "intercept": intercept,
            #"num_peaks": num_peaks,
            "cv": std / mean if mean != 0 else np.nan,
        }

def compute_ts_metrics_window(sub_df_lab):
    results = []

    grouped = sub_df_lab.groupby(["HADM_ID", "label"])
    for (hadm_id, label), group in tqdm(grouped, total=len(grouped)):
        group = group.sort_values("hour")
        times = group["hour"].values
        values = group["VALUENUM"].values

        # Whole 48h
        res_all = compute_ts_stats(times, values)
        res_all = {f"all_{k}": v for k, v in res_all.items()}

        # First 24h
        group_first = group[group["hour"] < 24]
        times_first = group_first["hour"].values
        values_first = group_first["VALUENUM"].values
        res_first = compute_ts_stats(times_first, values_first) if len(values_first) > 0 else {}
        res_first = {f"first24_{k}": v for k, v in res_first.items()}

        # Second 24h
        group_last = group[group["hour"] >= 24]
        times_last = group_last["hour"].values
        values_last = group_last["VALUENUM"].values
        res_last = compute_ts_stats(times_last, values_last) if len(values_last) > 0 else {}
        res_last = {f"last24_{k}": v for k, v in res_last.items()}

        # Differences if both available
        res_diff = {}
        for k in res_first:
            short_k = k.replace("first24_", "")
            if f"last24_{short_k}" in res_last:
                diff_val = res_last[f"last24_{short_k}"] - res_first[k]
                res_diff[f"delta_{short_k}"] = diff_val

        # Combine all
        result = {
            "HADM_ID": hadm_id,
            "label": label,
            "last_val": values[-1] if len(values) > 0 else np.nan,
            "first_val": values[0] if len(values) > 0 else np.nan,
            **res_all,
            **res_first,
            **res_last,
            **res_diff
        }

        results.append(result)

    return pd.DataFrame(results)

def compute_ts_metrics(sub_df_lab):
    results = []
    
    for (hadm_id, label), group in sub_df_lab.groupby(["HADM_ID", "label"]):
        group = group.sort_values("hour")  # assume hour is in ascending order
        times = group["hour"].values
        values = group["VALUENUM"].values
    
        # basic stats
        minimum = np.min(values)
        maximum = np.max(values)
        mean = np.mean(values)
        std = np.std(values)
        value_range = maximum - minimum
    
        # trend (slope from linear regression)
        try:
            if len(values) > 1:
                slope, intercept, _, _, _ = linregress(times, values)
            else:
                slope, intercept = np.nan, np.nan
        except ValueError:
            slope, intercept = np.nan, np.nan
    
        # pattern features
        peaks, _ = find_peaks(values)
        num_peaks = len(peaks)

        results.append({
            "HADM_ID": hadm_id,
            "label": label,
            "mean": mean,
            "std": std,
            "min": minimum,
            "max": maximum,
            "range": value_range,
            "slope": slope,
            "intercept": intercept,
            "num_peaks": num_peaks,
            "count": len(values),
            "cv": std / mean if mean != 0 else np.nan,
        })
    
    return pd.DataFrame(results)

