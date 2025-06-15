import os
import pandas as pd
import numpy as np
from itertools import chain
from tqdm import tqdm

# Define dataset name (used in file naming)
dataset = "mimiciii"

os.makedirs(f"{dataset}_all", exist_ok=True)
os.makedirs(f"{dataset}_last", exist_ok=True)


# Define the path to raw MIMIC-III CSV files
RAW_DATA_PATH = '../mimic-code/mimic-iii/buildmimic/postgres/MIMIC-III-data'  # <-- modify as needed

# dataset source https://physionet.org/content/mimiciii/1.4/

# Load lab item metadata (e.g., label descriptions)
lab_items = pd.read_csv(os.path.join(RAW_DATA_PATH, 'D_ITEMS.csv'))

# ------------------ Process admissions ------------------

# Load admission data
admissions_path = os.path.join(RAW_DATA_PATH, 'ADMISSIONS.csv')
admissions = pd.read_csv(admissions_path, usecols=[
    'HADM_ID', 'SUBJECT_ID', 'ADMISSION_TYPE', 'ADMITTIME', 'DISCHTIME', 'HOSPITAL_EXPIRE_FLAG'])

# Compute length of stay (LOS) in hours
admissions['ADMITTIME'] = pd.to_datetime(admissions['ADMITTIME'])
admissions['DISCHTIME'] = pd.to_datetime(admissions['DISCHTIME'])
admissions['LOS_HOURS'] = (admissions['DISCHTIME'] - admissions['ADMITTIME']).dt.total_seconds() / 3600

# Select relevant columns
outcomes = admissions[['HADM_ID', 'SUBJECT_ID', 'ADMISSION_TYPE', 'ADMITTIME',
                             'DISCHTIME', 'LOS_HOURS', 'HOSPITAL_EXPIRE_FLAG']].copy()

# Filter: only keep stays longer than 48 hours
outcomes = outcomes[outcomes["LOS_HOURS"] > 48]

# Add binary label: 1 if stay > 7 days (168 hours), else 0
outcomes["LONG_STAY"] = (outcomes["LOS_HOURS"] > 168).astype("int")

print(f"[Checkpoint] Admission data processed.")


# ------------------ Define relevant ITEMIDs ------------------

# Group relevant item IDs from CHARTEVENTS and OUTPUTEVENTS
itemids = {
    "SpO2": [646, 220277],
    "HR": [211, 220045],
    "RR": [618, 615, 220210, 224690],
    "SBP": [51, 442, 455, 6701, 220179, 220050],
    "DBP": [8368, 8440, 8441, 8555, 220180, 220051],
    "EtCO2": [1817, 228640],
    "Temp_F": [223761, 678],
    "Temp_C": [223762, 676],
    "TGCS": [198, 226755, 227013],
    "CRR": [3348, 115, 223951, 8377, 224308],
    "FiO2": [2981, 3420, 3422, 223835],
    "Glucose": [807, 811, 1529, 3745, 3744, 225664, 220621, 226537],
    "pH": [780, 860, 1126, 1673, 3839, 4202, 4753, 6003, 220274, 220734, 223830, 228243],
    "urine": [
        43647, 43053, 43171, 43173, 43333, 43347, 43348, 43355, 43365, 43373, 43374, 43379,
        43380, 43431, 43519, 43522, 43537, 43576, 43583, 43589, 43638, 43654, 43811, 43812,
        43856, 44706, 45304, 227519]
}

# Create reverse mapping and full ID list
itemids_inv = {v1: k for k, v in itemids.items() for v1 in v}
ids = list(chain.from_iterable(itemids.values()))
list_adm_id = outcomes["HADM_ID"].tolist()

# Paths to event files
chartevents_path = os.path.join(RAW_DATA_PATH, "CHARTEVENTS.csv")
outputevents_path = os.path.join(RAW_DATA_PATH, "OUTPUTEVENTS.csv")

# ------------------ Extract events data ------------------

# Read CHARTEVENTS in chunks and filter rows by HADM_ID and ITEMID
chunksize = 10**6
n_rows = 330712484  # total rows in CHARTEVENTS
n_chunks = n_rows // chunksize + int(n_rows % chunksize != 0)

df1 = []
for chunk in tqdm(pd.read_csv(chartevents_path,
                              usecols=["HADM_ID", "ITEMID", "CHARTTIME", "VALUENUM", "VALUEUOM", "VALUE"],
                              chunksize=chunksize),
                  total=n_chunks,
                  desc="Processing CHARTEVENTS"):
    filtered = chunk[(chunk["HADM_ID"].isin(list_adm_id)) & (chunk["ITEMID"].isin(ids))]
    df1.append(filtered)

df_labs = pd.concat(df1)

print(f"[Checkpoint] CHARTEVENTS data extracted.")

# Read OUTPUTEVENTS similarly
n_rows = 4349219
n_chunks = n_rows // chunksize + int(n_rows % chunksize != 0)

uo_itemids = itemids.get("urine", [])  # urine itemids from OUTPUTEVENTS

df2 = []
for chunk in tqdm(pd.read_csv(outputevents_path,
                              usecols=["HADM_ID", "ITEMID", "CHARTTIME", "VALUE"],
                              chunksize=chunksize),
                  total=n_chunks,
                  desc="Processing OUTPUTEVENTS"):
    filtered = chunk[(chunk["HADM_ID"].isin(list_adm_id)) & (chunk["ITEMID"].isin(uo_itemids))]
    df2.append(filtered)

print(f"[Checkpoint] OUTPUTEVENTS data extracted.")

# Combine CHARTEVENTS and OUTPUTEVENTS
df_out = pd.concat(df2)
df_labs = pd.concat([df_labs, df_out])

del df1, df2

# ------------------ Time filtering and feature processing ------------------

# Convert chart time to datetime
df_labs["CHARTTIME"] = pd.to_datetime(df_labs["CHARTTIME"])

# Merge with admission time and filter to first 48 hours
df_labs = df_labs.merge(outcomes[["HADM_ID", "ADMITTIME", "ADMISSION_TYPE"]], on="HADM_ID")
df_labs = df_labs[(df_labs["CHARTTIME"] - df_labs["ADMITTIME"]).dt.total_seconds() / 3600 <= 48]

# Compute relative time in minutes and hours
df_labs["minute"] = ((df_labs["CHARTTIME"] - df_labs["ADMITTIME"]).dt.total_seconds() / 60).astype("int")
df_labs["hour"] = (df_labs["minute"] / 60).astype("int")

# Add label for each ITEMID
df_labs["label"] = df_labs["ITEMID"].map(itemids_inv)

# Postprocess: fix specific value types (urine, temperature, TGCS, etc.)
mask = df_labs["label"] == "urine"
df_labs.loc[mask, "VALUENUM"] = df_labs.loc[mask, "VALUE"]

# Convert Fahrenheit to Celsius
mask = df_labs["label"] == "Temp_F"
df_labs.loc[mask, "VALUENUM"] = (df_labs.loc[mask, "VALUENUM"] - 32) / 1.8
df_labs.loc[mask, "label"] = "Temp_C"

# Map textual values for capillary refill
df_labs.loc[df_labs["VALUE"].isin(['Normal <3 secs', 'Normal <3 Seconds', 'Brisk']), "VALUENUM"] = 1
df_labs.loc[df_labs["VALUE"].isin(['Abnormal >3 secs', 'Abnormal >3 Seconds', 'Delayed']), "VALUENUM"] = 2
df_labs.loc[df_labs["VALUE"].isin(['Other/Remarks', 'Comment']), "VALUENUM"] = np.nan
df_labs.loc[df_labs["VALUE"].isna(), "VALUENUM"] = np.nan

# Count outliers and negative values as missing values
df_labs.loc[(df_labs["VALUENUM"] >= 500) | (df_labs["VALUENUM"] <= 0), "VALUENUM"] = np.nan

# Remove missing values
df_labs = df_labs[df_labs["VALUENUM"].notna()]
df_labs = df_labs[df_labs["minute"] >= 0]
print(f"[Checkpoint] Features processed.")

# ------------------ Final lab dataframe ------------------

# Compute mean value for each label, per admission and minute
labs_df = df_labs.groupby(['HADM_ID', 'ITEMID', 'label', 'minute', 'hour'])["VALUENUM"].mean().reset_index()

# Save labs data
labs_df.to_csv(f"{dataset}_all/all_labs.csv")  

print(f"[Checkpoint] Data saved for all admissions.")

# ------------------ Save targets ------------------

# Keep only admissions with available lab data
outcomes = outcomes[outcomes["HADM_ID"].isin(labs_df["HADM_ID"].unique())]
outcomes.to_csv(f"{dataset}_all/adm_target.csv")

# Extract last admission per patient
outcomes['ADMITTIME'] = pd.to_datetime(outcomes['ADMITTIME'])
outcomes = outcomes.sort_values(['SUBJECT_ID', 'ADMITTIME'])
last_stay = outcomes.drop_duplicates(subset='SUBJECT_ID', keep='last')

# Save last admissions and corresponding lab data
last_stay.reset_index(drop=True).to_csv(f"{dataset}_last/adm_target.csv")
labs_df[labs_df["HADM_ID"].isin(last_stay["HADM_ID"])].to_csv(f"{dataset}_last/all_labs.csv")

print(f"[Checkpoint] Data saved for last admissions.")


