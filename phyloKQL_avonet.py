
import os
import re
import glob
import uuid
import pickle
import argparse
import numpy as np
import pandas as pd

from phyloK.alpha_beta_weighting import (alpha_weighting,
                                               split_data_idx)
from phylokrr.treeio import parseNewickTree, is_leaf, writeNewickTree
try:
    from phyloK.tree_solve import get_training_tree
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


def response_columns(cols, response):
    """
    columns.csv rows -> (response_type, response columns)
    binary column      -> ('logistic', [column])
    categorical block  -> ('multinomial', [one-hot columns of that variable])
    """
    binary = list(cols.loc[cols['type'] == 'binary', 'column'])
    categorical = list(dict.fromkeys(cols.loc[cols['type'] == 'binary', 'variable']))
    if response in binary:
        return 'logistic', [response]
    if response in categorical:
        block = cols.loc[(cols['variable'] == response) & (cols['type'] == 'binary'), 'column']
        return 'multinomial', list(block)
    raise ValueError(f"--response must be a binary column {binary} "
                     f"or a categorical variable {categorical}")


def parse_merges(cols, response, merge):
    """
    --merge ["Scavenger=Carnivore", ...] -> [(source column, target column), ...]
    classes of the variable of the response (binary column or categorical variable)
    """
    if not merge:
        return [], []
    is_col = response in set(cols['column'])
    variable = cols.loc[cols['column'] == response, 'variable'].iloc[0] if is_col else response
    block = list(cols.loc[(cols['variable'] == variable) & (cols['type'] == 'binary'), 'column'])
    names = [c.split('=', 1)[1] for c in block]
    pairs = []
    for mg in merge:
        if mg.count('=') != 1:
            raise ValueError(f"--merge {mg!r}: use Source=Target, e.g. Scavenger=Carnivore")
        src, tgt = (s.strip() for s in mg.split('='))
        for s in (src, tgt):
            if s not in names:
                raise ValueError(f"--merge {mg!r}: {s!r} is not a class of {variable} {names}")
        if src == tgt:
            raise ValueError(f"--merge {mg!r}: source and target are the same class")
        pairs.append((f"{variable}={src}", f"{variable}={tgt}"))
    sources = [s for s, _ in pairs]
    if len(set(sources)) != len(sources) or set(sources) & {t for _, t in pairs}:
        raise ValueError(f"--merge {merge}: each class can be merged once and cannot be both source and target")
    return pairs, block


def main(x_csv, columns_csv, tree_file, response,
         data_model = 'avonet', ID = None, M1 = 1.0, M2 = 1.0,
         seed = 12038, test_num = 0.5, process = 'OU',
         SW = [1.0], folder_dir = 'out_dir', B = 1, verbose = True,
         min_class_count = 0, merge = None):

    if B < 1:
        raise ValueError("--B must be >= 1")
    cols = pd.read_csv(columns_csv)
    merges, block = parse_merges(cols, response, merge)
    if ID is None:
        ID = default_id(response) + "".join(
            f"_{s.split('=', 1)[1]}-into-{t.split('=', 1)[1]}" for s, t in merges)
    os.makedirs(folder_dir, exist_ok=True)

    # --- data ------------------------------------------------------------
    X_df = pd.read_csv(x_csv)
    predictors = list(cols.loc[cols['type'] != 'binary', 'column'])
    response_type, resp_cols = response_columns(cols, response)

    if merges and response_type == 'logistic':
        _, tgt_resp = next(((s, t) for s, t in merges if response in (s, t)), (None, None))
        if tgt_resp != response:
            raise ValueError(f"--merge {merge}: with a binary response, the target of the merge "
                             f"must be the response class ({response})")
    # with --merge, load the whole one-hot block of the variable (the merged classes are needed)
    load_cols = block if merges else resp_cols

    df = X_df[['species'] + load_cols + predictors]
    keep = df.notna().all(axis=1)
    if response_type == 'multinomial' or merges:
        keep &= (df[load_cols].sum(axis=1) == 1)   # exactly one class
    dropped = df.loc[~keep, 'species'].tolist()
    df = df[keep].set_index('species')
    print(f"{x_csv}: {len(X_df)} species | response {response} ({response_type}) | {len(predictors)} predictors")
    print(f"complete cases: {len(df)} (dropped {len(dropped)} with a missing response or predictor)")

    # --- merge classes ------------------------------------------------------
    for s, t in merges:
        n_src = int(df[s].sum())
        df[t] = np.maximum(df[t], df[s])
        df = df.drop(columns=[s])
        if s in resp_cols:
            resp_cols = [c for c in resp_cols if c != s]
        print(f"merged {s.split('=', 1)[1]} into {t.split('=', 1)[1]} ({n_src} species)")
    df = df[resp_cols + predictors]

    if response_type == 'multinomial':
        counts = df[resp_cols].sum(axis=0).astype(int)
        print("class counts:", {c.split('=', 1)[1]: int(v) for c, v in counts.items()})
        rare = [c for c in resp_cols if counts[c] < min_class_count]
        if rare:
            rows = df[rare].sum(axis=1) > 0
            dropped += df.index[rows].tolist()
            df = df[~rows]
            resp_cols = [c for c in resp_cols if c not in rare]
            print(f"dropped classes with < {min_class_count} species: "
                  f"{[c.split('=', 1)[1] for c in rare]} ({int(rows.sum())} species)")
        if len(resp_cols) < 2:
            raise ValueError("fewer than 2 classes left")
    classes = [c.split('=', 1)[1] for c in resp_cols]

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
    if response_type == 'logistic':
        y = df[response].to_numpy(dtype=float)          # (n,)
    else:
        y = df[resp_cols].to_numpy(dtype=float)         # (n, m) one-hot
    R = model_correlation(D, root, leaves, adj, C, process)
    freq = (f"P(y = 1) = {y.mean():.3f}" if response_type == 'logistic' else
            "class freq. " + ", ".join(f"{c} {v:.3f}" for c, v in zip(classes, y.mean(axis=0))))
    print(f"n = {len(y)} | p = {X.shape[1]} | {freq} | process {process} | t_max {t_max:.3f}")

    # --- splits, one dataset per split weight -----------------------------
    rng = np.random.default_rng(seed=seed)
    Indx = int(test_num * X.shape[0])
    c1 = c2 = 1.0

    datasets = {}
    for split_weight in SW:
        A = alpha_weighting(tree, alpha=split_weight, tol=1e-15, branch_length=False)
        W = np.array([A[leaf] for leaf in names])
        test_idx, train_idx = split_data_idx(Indx, W, rng)

        if response_type == 'logistic':
            train_ratio, test_ratio = y[train_idx].mean(), y[test_idx].mean()
            if verbose:
                print(f"\nsplit_weight: {split_weight} | train {len(train_idx)}, test {len(test_idx)} | "
                      f"label_ratio train {train_ratio:.3f}, test {test_ratio:.3f} | "
                      f"probs below 1e-8: {np.sum(W < 1e-8)}")
            if (not (0.2 < test_ratio < 0.8)) or (not (0.2 < train_ratio < 0.8)):
                print("Warning: unbalanced labels (outside 0.2-0.8); kept, unlike phyloKQL_simulate.py")
        else:
            n_tr, n_te = y[train_idx].sum(axis=0).astype(int), y[test_idx].sum(axis=0).astype(int)
            if verbose:
                print(f"\nsplit_weight: {split_weight} | train {len(train_idx)}, test {len(test_idx)} | "
                      f"probs below 1e-8: {np.sum(W < 1e-8)}")
                print("  class counts train / test: " +
                      ", ".join(f"{c} {a}/{b}" for c, a, b in zip(classes, n_tr, n_te)))
            empty = [c for c, a, b in zip(classes, n_tr, n_te) if a == 0 or b == 0]
            if empty:
                print(f"Warning: classes absent from the training or test set: {empty} "
                      f"(consider --min_class_count)")

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
            'classes': classes,          # column order of y (multinomial) / [positive class] (logistic)
        }

    meta = dict(data_model=data_model, seed=seed, test_num=test_num,
                ID=ID, M1=M1, M2=M2, tree_file=tree_out,
                process=process, data_process=None, SW=list(SW),
                N=None, C1=[c1], C2=[c2], B=B,
                n_trees=1, tree_idxs=[0]*len(datasets), tree_lines=[1]*len(datasets),
                empirical=True, x_csv=os.path.abspath(x_csv), source_tree=os.path.abspath(tree_file),
                response=response, predictors=predictors, dropped_species=dropped,
                response_type=response_type, classes=classes, response_columns=resp_cols,
                merged=[f"{s.split('=', 1)[1]}={t.split('=', 1)[1]}" for s, t in merges],
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
    print(f"Next: python phyloKQL_fit_v2.py -f {folder_dir} --out_dir {folder_dir} --data_model {data_model} "
          f"--ID {ID} --M1 {M1} --M2 {M2} --tree_file {tree_out}")
    print(f"      python phyloKQL_infer_v2.py -f {folder_dir} --data_model {data_model} --ID {ID} --M1 {M1} --M2 {M2}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='phyloKQL step 1/3 for AVONET: build the dataset pickle.')
    parser.add_argument('--x_csv', type=str, default='data_avonet_impute/X_obs_full.csv', help='X table from build_avonet_impute.py (default: data_avonet_impute/X_obs_full.csv)')
    parser.add_argument('--columns', type=str, default='data_avonet_impute/columns.csv', help='columns.csv from build_avonet_impute.py')
    parser.add_argument('--tree_file', type=str, default='data_avonet_impute/tree_full.nwk', help='Newick tree with the species as tips')
    parser.add_argument('--response', type=str, required=True, help='binary column (logistic), e.g. "Trophic.Level=Carnivore", or categorical variable (multinomial): Trophic.Level or Primary.Lifestyle')
    parser.add_argument('--merge', nargs='+', default=None, metavar='SOURCE=TARGET', help='relabel class SOURCE as TARGET, e.g. --merge Scavenger=Carnivore (default: none)')
    parser.add_argument('--min_class_count', type=int, default=0, help='multinomial only: drop the classes with fewer species than this (default: 0, keep all)')
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
         B=args.B, verbose=args.verbose, min_class_count=args.min_class_count,
         merge=args.merge)

"""
python phyloKQL_avonet.py --response "Trophic.Level=Carnivore" \
    --x_csv data_avonet_impute/X_true_500.csv --tree_file data_avonet_impute/tree_500.nwk \
    -f out_avonet --SW 1
python phyloKQL_fit.py   -f out_avonet --data_model avonet --ID Trophic.Level-Carnivore --M1 1 --M2 1
python phyloKQL_infer.py -f out_avonet --data_model avonet --ID Trophic.Level-Carnivore --M1 1 --M2 1

# multiclass (Trophic.Level: 4 classes, Primary.Lifestyle: 5 classes)
# scavengers as carnivores (3 classes; ID Trophic.Level_Scavenger-into-Carnivore):
#   python phyloKQL_avonet.py --response Trophic.Level --merge Scavenger=Carnivore ...
python phyloKQL_avonet.py --response Trophic.Level \
    --x_csv data_avonet_impute/X_true_500.csv --tree_file data_avonet_impute/tree_500.nwk \
    -f out_avonet --SW 1
python phyloKQL_fit_v2.py   -f out_avonet --out_dir out_avonet --data_model avonet --ID Trophic.Level --M1 1 --M2 1 \
    --tree_file out_avonet/tree_avonet_1.0_1.0_Trophic.Level.nwk
python phyloKQL_infer_v2.py -f out_avonet --data_model avonet --ID Trophic.Level --M1 1 --M2 1



# when merging (Scavenger=Carnivore)

python phyloKQL_avonet.py --response Trophic.Level \
--x_csv data_avonet_impute/X_true_500.csv --tree_file data_avonet_impute/tree_500.nwk \
-f out_avonet --SW 1 --merge Scavenger=Carnivore

python phyloKQL_fit_v2.py -f out_avonet --out_dir out_avonet --data_model avonet\
      --ID Trophic.Level_Scavenger-into-Carnivore --M1 1.0 --M2 1.0 --tree_file out_avonet/tree_avonet_1.0_1.0_Trophic.Level_Scavenger-into-Carnivore.nwk
      
python phyloKQL_infer_v2.py -f out_avonet --data_model avonet --ID Trophic.Level_Scavenger-into-Carnivore --M1 1.0 --M2 1.0
"""





# csv_file = "./data_avonet_impute/X_obs_full.csv"
# import pandas as pd

# df = pd.read_csv(csv_file)
# # take out rows with NAN on them
# df = df.dropna()


# for col in df.columns:
#     if '=' not in col:
#         continue

#     a = (df[col] == 1).sum()
#     print(col, a)


# sim_file = "./out_avonet/sim_avonet_1.0_1.0_Trophic.Level.pkl"
# with open(sim_file, 'rb') as f:
#     sim = pickle.load(f)
# a = list(sim['datasets'].values())
# a[0]