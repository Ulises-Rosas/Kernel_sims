"""
Extract the testing X and y of simulated datasets (sim_*.pkl from phyloKQL_simulate.py)
whose key (c1, c2, split_weight) matches --s1, --s2 and --sw.

Usage
    python get_test_set.py "out_dir/sim_*.pkl" --s1 0.5 --s2 2.0 --sw 1 [-o test_sets]

Writes, for every match, <out>/<pkl name>_X_test.csv and <pkl name>_y_test.csv
(rows = test taxa, first column = tip name).
"""
import os
import glob
import pickle
import argparse
import numpy as np
import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument('files', type=str, help='glob of the simulation files, e.g. "out_dir/sim_*.pkl"')
parser.add_argument('--s1', type=float, required=True, help='c1 (signal variance)')
parser.add_argument('--s2', type=float, required=True, help='c2 (noise variance)')
parser.add_argument('--sw', type=float, required=True, help='split weight (rho)')
parser.add_argument('-o', '--out_dir', type=str, default='test_sets')
args = parser.parse_args()

os.makedirs(args.out_dir, exist_ok=True)
query = np.array([args.s1, args.s2, args.sw])

n_found = 0
for f in sorted(glob.glob(args.files)):
    if f.endswith('_preMfix.pkl'):
        continue
    with open(f, 'rb') as fh:
        datasets = pickle.load(fh)['datasets']

    for key, ds in datasets.items():
        if not np.allclose(np.array(key, dtype=float), query):
            continue
        test_idx = ds['test_idx']
        X_test = ds['X'][test_idx, :]
        y_test = ds['y'][test_idx]
        tips = np.array(ds['tip_names'])[test_idx]

        base = os.path.join(args.out_dir, os.path.basename(f).replace('.pkl', ''))
        cols = [f"x{i + 1}" for i in range(X_test.shape[1])]
        pd.DataFrame(X_test, index=tips, columns=cols).to_csv(base + '_X_test.csv', index_label='tip')
        pd.DataFrame({'y': y_test}, index=tips).to_csv(base + '_y_test.csv', index_label='tip')

        n_found += 1
        print(f"{f}: key {key} -> X_test {X_test.shape}, y_test {y_test.shape} "
              f"(class 1: {np.mean(y_test == 1):.2f})")

if n_found == 0:
    print("No dataset matched; keys are (c1, c2, split_weight). Example keys from the last file:",
          list(datasets.keys())[:5] if 'datasets' in dir() else None)
