# Explaining medical time series classification through boundary-aware feature analysis

This repository contains code to identify key discriminative features in borderline ICU cases using interpretable time series summaries and constrained tree ensemble models. The approach focuses on admissions with highly similar temporal profiles but divergent outcomes, providing interpretable insights to support clinical decision-making and guideline refinement.

## 🔄 Pipeline Overview

1. **Data Preprocessing**
   - `1_preprocess_mimiciii.py`: Extracts and summarises lab time series data from MIMIC-III.
   - `1_preprocess_physionet.py`: Extracts and summarises data from the PhysioNet 2012 Challenge.

2. **Data Preparation**
   - `2_prepare_data.py`: Merges and formats time series summaries for downstream analysis.

3. **Pair Construction**
   - `3_compute_distances.py`: Computes pairwise distances between patient admissions.
   - `4_build_dataset_pairs.py`: Constructs admission pairs based on temporal similarity and outcome.

4. **Modelling**
   - `monotonic_tree.py`: Trains monotonic gradient boosting models to highlight discriminative features.
   - `utilities.py`: Contains utility functions for feature processing and rule extraction.

5. **Execution**
   - `bash.sh`: Shell script to run the full pipeline step-by-step.

## 📦 Requirements

- Python ≥ 3.8  
- Dependencies: `pandas`, `numpy`, `scikit-learn`, `xgboost`

Install required packages:

```bash
pip install -r requirements.txt
