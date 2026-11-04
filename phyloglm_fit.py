"""
Phylogenetic logistic regression baseline (R package phylolm, phyloglm) on the
datasets simulated by phyloKQL_simulate.py.

For every simulated dataset, the features are standardized with the training
statistics (as in phyloKQL_fit.py), phyloglm(y ~ x1 + ... + xp) is fitted on
the training species with the dataset's tree pruned to them, and the test
species are predicted with P(y = 1 | x) = plogis(x' beta) (phylolm has no
predict() method for phyloglm, so the prediction uses the fixed effects only).
"""
import os
import re
import csv
import pickle
import shutil
import argparse
import tempfile
import subprocess
import numpy as np


HERE = os.path.dirname(os.path.abspath(__file__))
R_SCRIPT = os.path.join(HERE, "phyloglm_fit.R")


def get_tag(data_model, M1, M2, ID):
    return f"{data_model}_{M1}_{M2}_{ID}"


def get_batch_suffix(batch):
    return "" if batch is None else f"_batch{batch}"


def read_trees(tree_file):
    """Same rules as phyloKQL_simulate.py: one tree per line, no blank/# lines."""
    with open(tree_file, 'r') as f:
        return [t for t in (l.strip() for l in f) if t and not t.startswith('#')]


def get_tip_names(ds, meta, cache):
    """Tip label of each row of X/y. Stored by phyloKQL_simulate.py; for older
    simulation files, recovered by parsing the tree as the simulation did."""
    if 'tip_names' in ds:
        return ds['tip_names']
    from phylokrr.treeio import parseNewickTree, is_leaf
    tree_idx = ds.get('tree_idx', 0)
    if tree_idx not in cache:
        trees = read_trees(meta['tree_file'])
        nodes, _ = parseNewickTree(trees[tree_idx])
        cache[tree_idx] = [n.name for n in nodes if is_leaf(n)]
    return cache[tree_idx]


def get_tree_text(ds, meta, cache):
    key = ('tree', ds.get('tree_idx', 0))
    if key not in cache:
        cache[key] = read_trees(meta['tree_file'])[ds.get('tree_idx', 0)]
    return cache[key]


def fit_sim_file(sim_file, work_dir, method, btol, log_alpha_bound, boot, rscript,
                 tree_file = None):
    """Export every dataset of one simulation file, run phyloglm_fit.R once and
    return {key: (y_test, probs)} and the per-dataset parameter rows."""

    with open(sim_file, 'rb') as f:
        sim = pickle.load(f)
    meta, datasets = sim['meta'], sim['datasets']
    keys = list(datasets.keys())
    if not keys:
        print(f"{sim_file}: no datasets")
        return {}, []

    if tree_file is not None:
        # tree given on the command line instead of the path stored in the .pkl
        # (that path is relative to the folder the simulation was run from)
        meta = dict(meta, tree_file=tree_file)
    elif not os.path.exists(meta['tree_file']):
        raise FileNotFoundError(f"tree file stored in {sim_file} not found: {meta['tree_file']} "
                                f"(it is relative to the folder the simulation was run from); "
                                f"pass it with --tree_file")

    cache = {}
    manifest = []
    for k, key in enumerate(keys):
        ds = datasets[key]
        X, y = ds['X'], ds['y']
        train_idx, test_idx = ds['train_idx'], ds['test_idx']
        tips = np.asarray(get_tip_names(ds, meta, cache))
        if tree_file is not None:
            tree_tips = set(re.findall(r"[(,]\s*([^():,;\s]+)\s*:", get_tree_text(ds, meta, cache)))
            if not set(tips) <= tree_tips:
                raise ValueError(f"{tree_file} (line {ds.get('tree_idx', 0) + 1}) lacks "
                                 f"{len(set(tips) - tree_tips)} species of {sim_file} {key}")
        if len(tips) != X.shape[0]:
            raise ValueError(f"{sim_file} {key}: {len(tips)} tip names for {X.shape[0]} rows")

        X_train = X[train_idx, :].copy()
        X_test = X[test_idx, :].copy()

        # same standardization as phyloKQL_fit.py / phyloKQL_infer.py
        X_train_offset = X_train.mean(axis=0)
        X_train_scale = X_train.std(axis=0)
        X_train = (X_train - X_train_offset) / X_train_scale
        X_test = (X_test - X_train_offset) / X_train_scale

        x_cols = [f"x{j+1}" for j in range(X.shape[1])]
        with open(os.path.join(work_dir, f"train_{k}.csv"), 'w', newline='') as f:
            w = csv.writer(f)
            w.writerow(['tip', 'y'] + x_cols)
            for tip, yi, xi in zip(tips[train_idx], y[train_idx], X_train):
                w.writerow([tip, int(yi)] + [repr(float(v)) for v in xi])
        

        with open(os.path.join(work_dir, f"test_{k}.csv"), 'w', newline='') as f:
            w = csv.writer(f)
            w.writerow(['tip'] + x_cols)
            for tip, xi in zip(tips[test_idx], X_test):
                w.writerow([tip] + [repr(float(v)) for v in xi])

        # get tree from working directory 
        with open(os.path.join(work_dir, f"tree_{k}.nwk"), 'w') as f:
            f.write(get_tree_text(ds, meta, cache) + "\n")

        manifest.append([k, f"train_{k}.csv", f"test_{k}.csv", f"tree_{k}.nwk"])

    with open(os.path.join(work_dir, "manifest.csv"), 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['k', 'train', 'test', 'tree'])
        w.writerows(manifest)

    print(f"{sim_file}: fitting {len(keys)} dataset(s) with phyloglm")
    subprocess.run([rscript, R_SCRIPT, work_dir, method, str(btol),
                    str(log_alpha_bound), str(boot)], check=True)

    with open(os.path.join(work_dir, "params.csv"), newline='') as f:
        r_params = {int(r['k']): r for r in csv.DictReader(f)}

    out, params = {}, []
    for k, key in enumerate(keys):
        ds = datasets[key]
        with open(os.path.join(work_dir, f"pred_{k}.csv"), newline='') as f:
            probs = [float(r['prob']) if r['prob'] not in ('NA', '') else np.nan
                     for r in csv.DictReader(f)]
        out[key] = (ds['y'][ds['test_idx']], np.asarray(probs))

        row = dict(r_params[k])
        row.pop('k')
        c1, c2, split_weight = key
        params.append(dict(c1=c1, c2=c2, split_weight=split_weight,
                           tree_line=ds.get('tree_line', ''), **row))
    return out, params


def main(data_model = 'friedman', ID = "test", M1 = 1.0, M2 = 1.0,
         folder_dir = 'out_dir', batch = None, out_dir = None,
         method = 'logistic_MPLE', btol = 10, log_alpha_bound = 4, boot = 0,
         rscript = 'Rscript', keep_tmp = False, tree_file = None):

    tag = get_tag(data_model, M1, M2, ID)
    path = lambda b: os.path.join(folder_dir, f"sim_{tag}{get_batch_suffix(b)}.pkl")

    # all outputs (and the temporary files exchanged with R) go to out_dir;
    # the simulation files are read from folder_dir
    out_dir = folder_dir if out_dir is None else out_dir
    os.makedirs(out_dir, exist_ok=True)

    # same batch logic as phyloKQL_infer.py
    if batch is not None:
        batches, out_suffix = [batch], get_batch_suffix(batch)
    elif os.path.exists(path(None)):
        batches, out_suffix = [None], ""
    else:
        if not os.path.exists(path(0)):
            raise FileNotFoundError(f"Neither {path(None)} nor {path(0)} found")
        with open(path(0), 'rb') as f:
            B = pickle.load(f)['meta']['B']
        batches, out_suffix = list(range(B)), ""

    out, params = {}, []
    for b in batches:
        work_dir = tempfile.mkdtemp(prefix=f"phyloglm_{tag}{get_batch_suffix(b)}_",
                                    dir=out_dir)
        try:
            o, p = fit_sim_file(path(b), work_dir, method, btol,
                                log_alpha_bound, boot, rscript, tree_file=tree_file)
        finally:
            if keep_tmp:
                print(f"Kept phyloglm inputs/outputs in {work_dir}")
            else:
                shutil.rmtree(work_dir, ignore_errors=True)
        out.update(o)
        params.extend(p)

    out_file = os.path.join(out_dir, f"summary_results_phyloglm_{tag}{out_suffix}.csv")
    with open(out_file, 'w') as f:
        for (c1, c2, split_weight), (y_test, probs) in out.items():
            f.write(','.join([str(c1), str(c2), str(split_weight), 'true'] + [str(p) for p in y_test]) + '\n')
            f.write(','.join([str(c1), str(c2), str(split_weight), 'phyloglm'] + [str(p) for p in probs]) + '\n')

    params_file = os.path.join(out_dir, f"phyloglm_params_{tag}{out_suffix}.csv")
    cols = list(dict.fromkeys(c for p in params for c in p))
    with open(params_file, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=cols, restval='NA')
        w.writeheader()
        w.writerows(params)

    for (c1, c2, split_weight), (y_test, probs) in out.items():
        acc = np.mean((probs >= 0.5) == y_test) if not np.isnan(probs).any() else np.nan
        print(f"c1: {c1} c2: {c2} split_weight: {split_weight} phyloglm acc: {acc}")

    status = [p.get('status') for p in params]
    print(f"phyloglm: {len(params)} fit(s): {status.count('ok')} ok, "
          f"{status.count('no_convergence')} without convergence, "
          f"{status.count('error')} failed (NA predictions)")
    print(f"Saved predictions to {out_file}")
    print(f"Saved parameters to {params_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Phylogenetic logistic regression (phylolm::phyloglm) on phyloKQL simulations.')
    parser.add_argument('--data_model', type=str, default='friedman', help='Data model used in the simulation (default: friedman)')
    parser.add_argument('--ID', type=str, default='test', help='Identifier for the experiment (default: test)')
    parser.add_argument('--M1', type=float, default=1.0, help='Parameter M1 (default: 1.0)')
    parser.add_argument('--M2', type=float, default=1.0, help='Parameter M2 (default: 1.0)')
    parser.add_argument('--folder_dir', '-f', type=str, default='out_dir', help='Directory with the simulation files (default: out_dir)')
    parser.add_argument('--out_dir', type=str, default=None, help='Directory for all outputs and the temporary R files (default: --folder_dir)')
    parser.add_argument('--batch', type=int, default=None, help='Fit only this simulation batch, 0..B-1 (default: all batches, combined into one output)')
    parser.add_argument('--method', type=str, default='logistic_MPLE', choices=['logistic_MPLE', 'logistic_IG10', 'logistic_MLE'], help='phyloglm method (default: logistic_MPLE)')
    parser.add_argument('--btol', type=float, default=10, help='phyloglm btol: bound on the linear predictor (default: 10)')
    parser.add_argument('--log_alpha_bound', type=float, default=4, help='phyloglm log.alpha.bound (default: 4)')
    parser.add_argument('--boot', type=int, default=0, help='phyloglm bootstrap replicates (default: 0)')
    parser.add_argument('--rscript', type=str, default='Rscript', help='Rscript executable (default: Rscript)')
    parser.add_argument('--keep_tmp', action='store_true', help='Keep the files exchanged with R (default: removed)')
    parser.add_argument('--tree_file', type=str, default=None, help='Tree file used to build the .pkl (one Newick per line), overriding the path stored in it (default: the stored path)')
    args = parser.parse_args()

    main(data_model=args.data_model, ID=args.ID, M1=args.M1, M2=args.M2,
         folder_dir=args.folder_dir, batch=args.batch, out_dir=args.out_dir, method=args.method,
         btol=args.btol, log_alpha_bound=args.log_alpha_bound, boot=args.boot,
         rscript=args.rscript, keep_tmp=args.keep_tmp, tree_file=args.tree_file)

"""
python phyloglm_fit.py -f out_dir --data_model friedman --ID test --M1 1 --M2 1
python phyloglm_fit.py -f out_dir --batch 3        # one batch (e.g. Slurm array task)
python phyloglm_fit.py -f out_avonet --ID Trophic.Level-Carnivore --data_model avonet

python phyloglm_fit.py -f out_avonet --ID Trophic.Level-Carnivore\
      --data_model avonet --tree_file out_avonet/tree_avonet_1.0_1.0_Trophic.Level-Carnivore.nwk
"""


# sim_file = "/Users/ulises/Desktop/MLab/Kernels/Kernel_sims/Kernel_sims/out_dir/sim_circle2_1.0_1.0_test.pkl"
# with open(sim_file, 'rb') as f:
#     sim = pickle.load(f)
# sim