# PHYSIONET

python 1_preprocess_physionet.py
python 2_prepare_data.py --dataset physionet --dept ALL --interval 1
python 2_prepare_data.py --dataset mimiciii_last --dept EMERGENCY --interval 1

python 3_compute_distances.py --dataset physionet --dept ALL --distance euclidean --interval 1
python 3_compute_distances.py --dataset physionet --dept ALL --distance chebyshev --interval 1
python 3_compute_distances.py --dataset physionet --dept ALL --distance cityblock --interval 1
python 3_compute_distances.py --dataset physionet --dept ALL --distance cosine --interval 1
python 3_compute_distances.py --dataset physionet --dept ALL --distance correlation --interval 1
python 3_compute_distances.py --dataset physionet --dept ALL --distance canberra --interval 1
python 3_compute_distances.py --dataset physionet --dept ALL --distance dtw --interval 1

python 4_build_dataset_pairs.py --dataset physionet --dept ALL --distance euclidean --computed imputed --pairs unmatched --features measured --MAX 3 --interval 1 --percentile 0.05
python 4_build_dataset_pairs.py --dataset physionet --dept ALL --distance chebyshev --computed imputed --pairs unmatched --features measured --MAX 3 --interval 1 --percentile 0.05
python 4_build_dataset_pairs.py --dataset physionet --dept ALL --distance cityblock --computed imputed --pairs unmatched --features measured --MAX 3 --interval 1 --percentile 0.05
python 4_build_dataset_pairs.py --dataset physionet --dept ALL --distance cosine --computed imputed --pairs unmatched --features measured --MAX 3 --interval 1 --percentile 0.05
python 4_build_dataset_pairs.py --dataset physionet --dept ALL --distance correlation --computed imputed --pairs unmatched --features measured --MAX 3 --interval 1 --percentile 0.05
python 4_build_dataset_pairs.py --dataset physionet --dept ALL --distance canberra --computed imputed --pairs unmatched --features measured --MAX 3 --interval 1 --percentile 0.05
python 4_build_dataset_pairs.py --dataset physionet --dept ALL --distance dtw --computed imputed --pairs unmatched --features measured --MAX 3 --interval 1 --percentile 0.05

# MIMIC III

python 1_preprocess_mimiciii.py
python 2_prepare_data.py --dataset mimiciii_last --dept EMERGENCY --interval 1

python 3_compute_distances.py --dataset mimiciii_last --dept EMERGENCY --distance euclidean --interval 1
python 3_compute_distances.py --dataset mimiciii_last --dept EMERGENCY --distance chebyshev --interval 1
python 3_compute_distances.py --dataset mimiciii_last --dept EMERGENCY --distance cityblock --interval 1
python 3_compute_distances.py --dataset mimiciii_last --dept EMERGENCY --distance cosine --interval 1
python 3_compute_distances.py --dataset mimiciii_last --dept EMERGENCY --distance correlation --interval 1
python 3_compute_distances.py --dataset mimiciii_last --dept EMERGENCY --distance canberra --interval 1
python 3_compute_distances.py --dataset mimiciii_last --dept EMERGENCY --distance dtw --interval 1

python 4_build_dataset_pairs.py --dataset mimiciii_last --dept EMERGENCY --distance euclidean --computed imputed --pairs unmatched --features measured --MAX 3 --interval 1 --percentile 0.05
python 4_build_dataset_pairs.py --dataset mimiciii_last --dept EMERGENCY --distance chebyshev --computed imputed --pairs unmatched --features measured --MAX 3 --interval 1 --percentile 0.05
python 4_build_dataset_pairs.py --dataset mimiciii_last --dept EMERGENCY --distance cityblock --computed imputed --pairs unmatched --features measured --MAX 3 --interval 1 --percentile 0.05
python 4_build_dataset_pairs.py --dataset mimiciii_last --dept EMERGENCY --distance cosine --computed imputed --pairs unmatched --features measured --MAX 3 --interval 1 --percentile 0.05
python 4_build_dataset_pairs.py --dataset mimiciii_last --dept EMERGENCY --distance correlation --computed imputed --pairs unmatched --features measured --MAX 3 --interval 1 --percentile 0.05
python 4_build_dataset_pairs.py --dataset mimiciii_last --dept EMERGENCY --distance canberra --computed imputed --pairs unmatched --features measured --MAX 3 --interval 1 --percentile 0.05
python 4_build_dataset_pairs.py --dataset mimiciii_last --dept EMERGENCY --distance dtw --computed imputed --pairs unmatched --features measured --MAX 3 --interval 1 --percentile 0.05


