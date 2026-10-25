"""
Step 1/3 of the phyloKQL pipeline for EMPIRICAL data (AVONET), the
counterpart of phyloKQL_simulate.py.

Reads the output of data_avonet_impute/build_avonet_impute.py

    X csv       species + columns (X_obs_full.csv or X_true_500.csv)
    columns.csv column metadata: column, variable, type (continuous / ordinal / binary), transform
    tree        Newick tree whose tips are the species (tree_full.nwk or tree_500.nwk)

and writes, in the same format as phyloKQL_simulate.py,

    {folder_dir}/sim_{data_model}_{M1}_{M2}_{ID}.pkl                    (--B 1)
    {folder_dir}/sim_{data_model}_{M1}_{M2}_{ID}_batch{i}.pkl, i=0..B-1  (--B > 1)
    {folder_dir}/tree_{data_model}_{M1}_{M2}_{ID}.nwk    tree of the species kept (meta['tree_file'])

which phyloKQL_fit.py, phyloKQL_infer.py and phyloglm_fit.py read unchanged.

  * response  : one binary column of columns.csv, e.g. --response "Trophic.Level=Carnivore"
                -> y = 1 for that class, 0 for the rest
  * predictors: every non-binary column (continuous and ordinal), raw values
                (phyloKQL_fit.py standardizes with training statistics)
  * rows with a missing response or predictor are dropped (complete case);
    the tree is pruned to the species kept
  * R (model correlation input) and the train/test split are built exactly as in
    phyloKQL_simulate.py: R = 2*D/max(D) for OU, split with alpha_weighting(tree,
    alpha=split_weight) + split_data_idx, one dataset per --SW value.

M1 / M2 / data_model only name the files (kept for compatibility with the
other pipeline scripts); there is nothing to simulate.
"""
import os
import re
import glob
import uuid
import pickle
import argparse
import numpy as np
import pandas as pd

from phylokrr_dev.alpha_beta_weighting import (alpha_weighting,
                                               split_data_idx)
from phylokrr.treeio import parseNewickTree, is_leaf, writeNewickTree
try:
    from phylokrr_dev.tree_solve import get_training_tree
except ImportError:  # local development copy
    from tree_solve_dev import get_training_tree

# same tree utilities as the simulation, so R and the split match it exactly
from phyloKQL_simulate import get_D, get_R, process_D, get_tag, get_batch_suffix


def model_correlation(D, root, leaves, adj, C, process):
    """R used by the model; same as phyloKQL_simulate.simulate_data"""
    n = D.shape[0]
    if process == 'OU':
        return 2*D/np.max(D)                      # scaled path lengths
    elif process == 'BM':
        Ds = 2*D/np.max(D)
        V0 = process_D(Ds, root, leaves, adj, C, sigma2=1, alpha=1, process='BM')
        return get_R(V0)                          # BM correlation
    elif process == 'none':
        return np.eye(n)
    raise ValueError("process must be 'OU', 'BM' or 'none'")


def default_id(response):
    return re.sub(r"[^A-Za-z0-9._-]+", "-", response)


def main(x_csv, columns_csv, tree_file, response,
         data_model = 'avonet', ID = None, M1 = 1.0, M2 = 1.0,
         seed = 12038, test_num = 0.5, process = 'OU',
         SW = [1.0], folder_dir = 'out_dir', B = 1, verbose = True):

    if B < 1:
        raise ValueError("--B must be >= 1")
    ID = default_id(response) if ID is None else ID
    os.makedirs(folder_dir, exist_ok=True)

    # --- data ------------------------------------------------------------
    X_df = pd.read_csv(x_csv)
    cols = pd.read_csv(columns_csv)
    binary = list(cols.loc[cols['type'] == 'binary', 'column'])
    predictors = list(cols.loc[cols['type'] != 'binary', 'column'])
    if response not in binary:
        raise ValueError(f"--response must be one of the binary columns: {binary}")

    df = X_df[['species', response] + predictors]
    keep = df.notna().all(axis=1)
    dropped = df.loc[~keep, 'species'].tolist()
    df = df[keep].set_index('species')
    print(f"{x_csv}: {len(X_df)} species | response {response} | {len(predictors)} predictors")
    print(f"complete cases: {len(df)} (dropped {len(dropped)} with a missing response or predictor)")

    # --- tree: prune to the species kept ----------------------------------
    with open(tree_file) as f:
        tree = f.read().strip()
    nodes, _ = parseNewickTree(tree)
    tips = [n.name for n in nodes if is_leaf(n)]
    missing_in_tree = sorted(set(df.index) - set(tips))
    if missing_in_tree:
        raise ValueError(f"{len(missing_in_tree)} species are not tips of {tree_file}, e.g. {missing_in_tree[:3]}")
    if set(tips) != set(df.index):
        _, new_root = get_training_tree(nodes, set(df.index))
        tree = writeNewickTree(new_root)
        print(f"tree pruned from {len(tips)} to {len(df)} tips")

    tag = get_tag(data_model, M1, M2, ID)
    tree_out = os.path.join(folder_dir, f"tree_{tag}.nwk")
    with open(tree_out, 'w') as f:
        f.write(tree + "\n")

    D, root, leaves, adj, C = get_D(tree)
    names = [n.name for n in leaves]
    assert set(names) == set(df.index), "pruned tree tips differ from the data"
    t_max = np.max(D)

    # rows in the tip order of the tree (= row order of D and R)
    df = df.loc[names]
    X = df[predictors].to_numpy(dtype=float)
    y = df[response].to_numpy(dtype=float)
    R = model_correlation(D, root, leaves, adj, C, process)
    print(f"n = {len(y)} | p = {X.shape[1]} | P(y = 1) = {y.mean():.3f} | process {process} | t_max {t_max:.3f}")

    # --- splits, one dataset per split weight -----------------------------
    rng = np.random.default_rng(seed=seed)
    Indx = int(test_num * X.shape[0])
    c1 = c2 = 1.0

    datasets = {}
    for split_weight in SW:
        A = alpha_weighting(tree, alpha=split_weight, tol=1e-15, branch_length=False)
        W = np.array([A[leaf] for leaf in names])
        test_idx, train_idx = split_data_idx(Indx, W, rng)

        train_ratio, test_ratio = y[train_idx].mean(), y[test_idx].mean()
        if verbose:
            print(f"\nsplit_weight: {split_weight} | train {len(train_idx)}, test {len(test_idx)} | "
                  f"label_ratio train {train_ratio:.3f}, test {test_ratio:.3f} | "
                  f"probs below 1e-8: {np.sum(W < 1e-8)}")
        if (not (0.2 < test_ratio < 0.8)) or (not (0.2 < train_ratio < 0.8)):
            print("Warning: unbalanced labels (outside 0.2-0.8); kept, unlike phyloKQL_simulate.py")

        datasets[(c1, c2, split_weight)] = {
            'X': X,
            'y': y,
            'train_idx': train_idx,
            'test_idx': test_idx,
            'W': W,
            # 'vcv_train': R[train_idx, :][:, train_idx],
            'vcv_train': R[np.ix_(train_idx, train_idx)],
            # 'vcv_test_12': R[test_idx, :][:, train_idx],
            'vcv_test_12': R[np.ix_(test_idx, train_idx)],
            'tree_idx': 0,
            'tree_line': 1,
            'tip_names': names,          # row order of X, y and R
        }

    meta = dict(data_model=data_model, seed=seed, test_num=test_num,
                ID=ID, M1=M1, M2=M2, tree_file=tree_out,
                process=process, data_process=None, SW=list(SW),
                N=None, C1=[c1], C2=[c2], B=B,
                n_trees=1, tree_idxs=[0]*len(datasets), tree_lines=[1]*len(datasets),
                empirical=True, x_csv=os.path.abspath(x_csv), source_tree=os.path.abspath(tree_file),
                response=response, predictors=predictors, dropped_species=dropped,
                t_max=float(t_max),
                sim_uid=uuid.uuid4().hex)  # lets fit/infer detect stale files

    # same file layout as phyloKQL_simulate.py (and removal of stale layouts)
    for old in ([os.path.join(folder_dir, f"sim_{tag}.pkl")] +
                glob.glob(os.path.join(folder_dir, f"sim_{tag}_batch*.pkl"))):
        if os.path.exists(old):
            os.remove(old)

    keys = list(datasets.keys())
    if B > len(keys):
        print(f"Warning: --B {B} > number of datasets ({len(keys)}); some batch files will be empty")
    for b, idx in enumerate(np.array_split(np.arange(len(keys)), B)):
        batch = None if B == 1 else b
        out_file = os.path.join(folder_dir, f"sim_{tag}{get_batch_suffix(batch)}.pkl")
        with open(out_file, 'wb') as f:
            pickle.dump({'meta': dict(meta, batch=batch),
                         'datasets': {keys[i]: datasets[keys[i]] for i in idx}}, f)
        print(f"Saved {len(idx)} dataset(s) to {out_file}")
    print(f"Tree of the kept species: {tree_out}")
    print(f"Next: python phyloKQL_fit.py -f {folder_dir} --data_model {data_model} --ID {ID} --M1 {M1} --M2 {M2}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='phyloKQL step 1/3 for AVONET: build the dataset pickle.')
    parser.add_argument('--x_csv', type=str, default='data_avonet_impute/X_obs_full.csv', help='X table from build_avonet_impute.py (default: data_avonet_impute/X_obs_full.csv)')
    parser.add_argument('--columns', type=str, default='data_avonet_impute/columns.csv', help='columns.csv from build_avonet_impute.py')
    parser.add_argument('--tree_file', type=str, default='data_avonet_impute/tree_full.nwk', help='Newick tree with the species as tips')
    parser.add_argument('--response', type=str, required=True, help='binary column used as y, e.g. "Trophic.Level=Carnivore"')
    parser.add_argument('--data_model', type=str, default='avonet', help='name used in the output files (default: avonet)')
    parser.add_argument('--ID', type=str, default=None, help='Identifier for the experiment (default: the response, e.g. Trophic.Level-Carnivore)')
    parser.add_argument('--M1', type=float, default=1.0, help='only used in file names (default: 1.0)')
    parser.add_argument('--M2', type=float, default=1.0, help='only used in file names (default: 1.0)')
    parser.add_argument('-p', '--process', type=str, default='OU', help='OU, BM or none (default: OU)')
    parser.add_argument('--seed', type=int, default=12038, help='Random seed for the split (default: 12038)')
    parser.add_argument('--test_num', type=float, default=0.5, help='Proportion of test data (default: 0.5)')
    parser.add_argument('--SW', nargs='+', type=float, default=[1.0], help='List of split weights (default: [1.0])')
    parser.add_argument('--folder_dir', '-f', type=str, default='out_dir', help='Output directory (default: out_dir)')
    parser.add_argument('--B', type=int, default=1, help='Number of batch files (default: 1)')
    parser.add_argument('-v', '--verbose', action=argparse.BooleanOptionalAction, default=True, help='Verbose output')
    args = parser.parse_args()

    main(x_csv=args.x_csv, columns_csv=args.columns, tree_file=args.tree_file, response=args.response,
         data_model=args.data_model, ID=args.ID, M1=args.M1, M2=args.M2, seed=args.seed,
         test_num=args.test_num, process=args.process, SW=args.SW, folder_dir=args.folder_dir,
         B=args.B, verbose=args.verbose)

"""
python phyloKQL_avonet.py --response "Trophic.Level=Carnivore" \
    --x_csv data_avonet_impute/X_true_500.csv --tree_file data_avonet_impute/tree_500.nwk \
    -f out_avonet --SW 1
python phyloKQL_fit.py   -f out_avonet --data_model avonet --ID Trophic.Level-Carnivore --M1 1 --M2 1
python phyloKQL_infer.py -f out_avonet --data_model avonet --ID Trophic.Level-Carnivore --M1 1 --M2 1
"""
