"""
Step 2/3 of the phyloKQL pipeline: FITTING.

Version 2: with --tree_file, the tree is used during cross-validation and
inference (fast tree-based solve of the correction, see test_phyloKQL.py)
"""
import os
import sys
import copy
import time
import pickle
import argparse
import numpy as np

from phylokrr.treeio import parseNewickTree, is_leaf
try:
    from phyloK.tree_solve import get_training_tree, clone_nodes
except ImportError:  # local development copy, as in test_phyloKQL.py
    from tree_solve_dev import get_training_tree, clone_nodes

import importlib

from phyloK.block_cross_validation import k_fold_cv_weigthed

from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

# response type -> model class name (in --model_module)
RESPONSE_TYPES = {
    'logistic'   : 'Logistic',
    'multinomial': 'Multinomial',
    'continuous' : 'PhyloKernel',
}


def get_model_class(model_module, response_type):
    """Logistic / Multinomial / PhyloKernel from e.g. phyloK_dev2 or phyloK.phyloK_v2"""
    module = importlib.import_module(model_module)
    return getattr(module, RESPONSE_TYPES[response_type])

# m = get_model_class("phyloK.phyloK_v2", 'logistic')
# a = m()
# a.reg

def check_response(y, response_type):
    """shape / values of y must match the response type"""
    if response_type == 'multinomial':
        if y.ndim != 2 or y.shape[1] < 2 or not np.all(np.isin(y, (0, 1))) or not np.all(y.sum(axis=1) == 1):
            raise ValueError("multinomial needs a one-hot y of shape (n, m), m >= 2")
    elif y.ndim != 1:
        raise ValueError(f"{response_type} needs a 1-D y, got shape {y.shape}")
    elif response_type == 'logistic' and not np.all(np.isin(y, (0, 1))):
        raise ValueError("logistic needs a binary (0/1) y")


def fit_baselines(response_type, X_train, y_train, seed):
    """non-phylogenetic baselines; classifiers use the class labels"""
    if response_type == 'continuous':
        lin_reg = LinearRegression()
        rf_model = RandomForestRegressor(n_estimators=100, random_state=seed)
        lin_reg.fit(X_train, y_train)
        start = time.time()
        rf_model.fit(X_train, y_train)
        print(f'[rf] fit Time: ', time.time() - start)
        return {'lin_reg': lin_reg, 'rf': rf_model}

    labels = y_train if response_type == 'logistic' else np.argmax(y_train, axis=1)
    log_reg = LogisticRegression() if response_type == 'logistic' else LogisticRegression(max_iter=1000)
    rf_model = RandomForestClassifier(n_estimators=100, random_state=seed)
    log_reg.fit(X_train, labels)
    start = time.time()
    rf_model.fit(X_train, labels)
    print(f'[rf] fit Time: ', time.time() - start)
    return {'log_reg': log_reg, 'rf': rf_model}


def get_tag(data_model, M1, M2, ID):
    return f"{data_model}_{M1}_{M2}_{ID}"


def get_batch_suffix(batch):
    return "" if batch is None else f"_batch{batch}"

# Model variants fitted on every simulated replicate. Each one is defined as a
# change with respect to the "final" model, i.e. the configuration given by
# the CLI flags (whose defaults are the final model).
VARIANTS = {
    'final'   : {},                                        # -p OU + corr + adj + scale + AGOP
    'no_tree' : {'process': 'none', 'do_corr': False, 'do_adj': None},# -p none (R = identity)
    'no_agop' : {'do_agop': False, 'agop_diag': False},    # without AGOP
    'no_corr' : {'do_corr': False, 'do_adj': None},        # --no-do_corr, no --do_adj
    'no_scale': {'corr_scale_error': None},                # without --corr_scale_error
}


def get_params(do_adj):
    # # ref: Kernel methods in computational biology, pp. 62
    if not do_adj:
        params = {
            'lam' : np.logspace(-12, 3, 100, base=2),
            'gamma'  : np.logspace(-15, 4, 100, base=2),
        }
    else:
        params = {
            'lam' : np.logspace(-12, 3, 100, base=2),
            'gamma'  : np.logspace(-15, 4, 100, base=2),
            'corr' : np.logspace(-10, 0, 100, base=2)
        }
    return params


def read_trees(tree_file):
    """One Newick tree per line; skips blank lines and '#' comments (as phyloKQL_simulate.py)."""
    with open(tree_file, 'r') as f:
        trees = [t for t in (l.strip() for l in f) if t and not t.startswith('#')]
    if not trees:
        raise ValueError(f"No trees found in {tree_file}")
    return trees


def tree_height(nodes):
    """root-to-tip path length of the first leaf (the tree is ultrametric)"""
    leaves = [node for node in nodes if is_leaf(node)]
    height = 0
    l = leaves[0]
    while l.ancestor is not None:
        height += l.branch_length
        l = l.ancestor
    return height


def scaled_tree(newick, t_max = None):
    """
    Parse a tree and scale its branch lengths to the scale of (2/t_max) * D,
    i.e. by 1/height with height = t_max/2 (estimated from the tree if t_max is None).
    """
    nodes, root = parseNewickTree(newick)
    height = t_max/2 if t_max is not None else tree_height(nodes)
    for n in nodes:
        n.branch_length = n.branch_length * (1/height)
    return nodes, root, height


def fit_lkrr(cfg, X_train, y_train, vcv_train, W_train,
             seed, cores, verbose, folder_dir, name, batch_suffix = "",
             also_no_agop = False, tree_train = None, spps_train = None,
             model_class = None, response_type = 'logistic'):
    """CV + refit of one phylo kernel model (model_class: Logistic, Multinomial
    or PhyloKernel) with configuration cfg.

    If also_no_agop, the model is first fitted without AGOP (that copy is
    returned as the no_agop model) and then with AGOP, sharing the same CV.

    tree_train (nodes, root) / spps_train: training tree, already pruned to the
    training species and scaled, and the training species (row order of
    X_train). If given, they are passed to the CV and set in the model before
    the final fit, for the tree-based solve at inference.
    """

    fix_phi = True
    cv_sample = 100 # number of hyperparameters to test
    cv_folds = 5
    max_iter = 100
    nested_loop_verbose = True

    if cfg['process'] == 'none':
        # ignore the tree regardless of how the data were simulated
        vcv_train = np.eye(vcv_train.shape[0])

    model_kw = dict(tol=cfg['tol'], max_iter=max_iter, kernel='rbf',
                    verbose=not False, fix_phi = fix_phi)
    if response_type != 'multinomial':
        model_kw['get_corr'] = False # Multinomial has no get_corr argument
    lkrr = model_class(**model_kw)
    lkrr.add_extraK = True # backward compatibility with previous versions of phyloKQL
    lkrr.process = cfg['process']
    lkrr.use_quasi_newton = cfg['use_qn']
    lkrr.do_corr = cfg['do_corr']
    lkrr.corr_type = cfg['do_adj'] # ignore if do_corr is False, since it is not possible to adjust the latent space without correction during CV
    lkrr.corr_scale_error = cfg['corr_scale_error']
    lkrr.agop_iter = cfg['agop_iter']
    lkrr.agop_nogreedy = cfg['agop_nogreedy']

    # this keeps time complexity
    # bellow n^2, but still adaptive enough
    # lkrr.memory_qn = np.sqrt(X_train.shape[0]).astype(int)
    lkrr.memory_qn = X_train.shape[0]

    # training tree information for the CV (only used with the correction)
    tree_kw = {} if tree_train is None else dict(nodes = tree_train[0], leaves = spps_train)

    start = time.time()
    lkrr_best_params = k_fold_cv_weigthed(
        X_train, y_train, vcv_train,
        lkrr,  W_train, get_params(cfg['do_adj']),
        folds=cv_folds,
        sample=cv_sample,
        verbose= verbose,
        seed=seed,
        cores = cores,
        **tree_kw
    )
    print(f'[{name}] lkrr CV Time: ', time.time() - start)
    if nested_loop_verbose:
        print(f'[{name}] lkrr best params: ', lkrr_best_params)

    lkrr.set_params(**lkrr_best_params)

    # re-set the training tree for the final fit and the inference
    if tree_train is not None:
        lkrr.set_training_tree(tree_train, spps_train)

    if nested_loop_verbose:
        print(f'[{name}] ADDITIVE APPROACH:')

    lkrr_no_agop = None
    if also_no_agop:
        # 1) fit without AGOP and keep a copy of that model
        lkrr.do_agop = False
        lkrr.agop_diag = False
        start = time.time()
        lkrr.fit(X_train, y_train, vcv_train)
        print(f'[no_agop] lkrr fit Time: ', time.time() - start)

        # copy of the model without the training tree (deepcopy would recurse
        # through every linked node); the no_agop copy gets its own clone of
        # the tree from clone_nodes (non-recursive), ready for inference
        tree_attr = getattr(lkrr, 'training_tree', None)
        lkrr.training_tree = None
        lkrr_no_agop = copy.deepcopy(lkrr)
        lkrr.training_tree = tree_attr
        if tree_attr is not None:
            cloned_root, cloned_nodes = clone_nodes(tree_attr[0])
            lkrr_no_agop.training_tree = (cloned_nodes, cloned_root)   # same (nodes, root) layout

    # 2) fit with this variant's AGOP settings
    lkrr.do_agop = cfg['do_agop'] # only applied when we know the hyperparameters.
    lkrr.agop_diag = cfg['agop_diag']
    start = time.time()
    lkrr.fit(X_train, y_train, vcv_train)
    print(f'[{name}] lkrr fit Time: ', time.time() - start)

    print(f'[{name}] agop M:')
    if lkrr.M is not None:
        print(np.round(lkrr.M, 3))

        # final model keeps the original file name
        agop_file = (f"agop_M{batch_suffix}.csv" if name == 'final'
                     else f"agop_M_{name}{batch_suffix}.csv")
        with open(f"{folder_dir}/{agop_file}", 'w') as f:
            # write the matrix M to a csv file
            np.savetxt(f, lkrr.M, delimiter=",")

    if nested_loop_verbose:
        print(f'[{name}] alpha param:', np.exp(-lkrr.psc))
        print(f'[{name}] signal (meaningless if process is not OU):',lkrr.psc)

    return lkrr, lkrr_best_params, lkrr_no_agop


def main(data_model = 'friedman', seed = 12038, ID = "test", M1 = 1.0, M2 = 1.0,
         process = 'OU',
         cores = 5, verbose = True, tol = 1e-6,
         use_qn = True, folder_dir = 'in_dir',
         do_adj = 'shrink', # adjust the level of correction for the latent space,
         do_corr = True, # do correction during cv,
         corr_scale_error = 'auto', # 'auto': std for logistic/multinomial, None for continuous. Options: [None, 'auto', 'std', 'n', 'neff']
         do_agop = True, # use average gradient outer product
         agop_iter = 5, # number of iterations for the average gradient outer product
         agop_diag = True, # use diagonal approximation for the average gradient outer product
         agop_nogreedy = False, # use greedy approach for the average gradient outer product
         variants = tuple(VARIANTS), # which model variants to fit (see VARIANTS)
         batch = None, # simulation batch to fit (None -> unbatched simulation, --B 1)
         out_dir = 'out_dir',
         tree_file = None, # tree(s) for CV and inference with the tree-based solve
         response_type = None, # logistic, multinomial or continuous (None -> meta of the .pkl, else logistic)
         model_module = 'phyloK_dev2', # module with the Logistic / Multinomial / PhyloKernel classes
         ):

    tag = get_tag(data_model, M1, M2, ID)
    batch_suffix = get_batch_suffix(batch)
    sim_file = os.path.join(folder_dir, f"sim_{tag}{batch_suffix}.pkl")
    if not os.path.exists(sim_file):
        raise FileNotFoundError(f"{sim_file} not found; if the simulation was "
                                f"run with --B > 1, pass --batch i (0-based)")

    # sim_file = "/Users/ulises/Desktop/MLab/Kernels/Kernel_sims/Kernel_sims/out_dir/sim_circle2_1.0_1.0_test.pkl"
    with open(sim_file, 'rb') as f:
        sim = pickle.load(f)
    meta, datasets = sim['meta'], sim['datasets']

    if seed is None:
        seed = meta['seed']

    # response type: CLI > simulation metadata > logistic
    sim_response_type = meta.get('response_type')
    if response_type is None:
        response_type = sim_response_type or 'logistic'
    elif sim_response_type is not None and response_type != sim_response_type:
        raise ValueError(f"--response_type {response_type} does not match {sim_file} ({sim_response_type})")
    if response_type not in RESPONSE_TYPES:
        raise ValueError(f"--response_type must be one of {list(RESPONSE_TYPES)}")
    model_class = get_model_class(model_module, response_type)
    print(f"Response type: {response_type} -> {model_module}.{model_class.__name__}")

    if corr_scale_error == 'auto':
        corr_scale_error = None if response_type == 'continuous' else 'std'

    base_cfg = dict(process=process, tol=tol, use_qn=use_qn,
                    do_corr=do_corr, do_adj=do_adj,
                    corr_scale_error=corr_scale_error, do_agop=do_agop,
                    agop_iter=agop_iter, agop_diag=agop_diag,
                    agop_nogreedy=agop_nogreedy)

    configs = {}
    for name in variants:
        if name not in VARIANTS:
            raise ValueError(f"Unknown variant '{name}'. Options: {list(VARIANTS)}")
        cfg = dict(base_cfg, **VARIANTS[name])

        if not cfg['do_corr'] and cfg['do_adj']:
            raise ValueError(f"[{name}] Cannot adjust latent space without correction during CV")

        # The tree correlation matrix R was built in the simulation step
        # for a given process, so the model must use that same process,
        # except for 'none', which ignores the tree (R = identity) regardless
        # of how the data were simulated.
        if cfg['process'] is None:
            cfg['process'] = meta['process']

        elif cfg['process'] != 'none' and cfg['process'] != meta['process']:
            raise ValueError(f"[{name}] --process {cfg['process']} does not match the process "
                             f"used to simulate {sim_file} ({meta['process']}); "
                             f"only '{meta['process']}' or 'none' are allowed")
        configs[name] = cfg

    # no_scale only differs from final if the final model scales the error
    if 'no_scale' in configs and 'final' in configs and configs['final']['corr_scale_error'] is None:
        print("no_scale skipped: the final model does not scale the correction error (same model)")
        del configs['no_scale']

    print('Model variants to fit:')
    for name, cfg in configs.items():
        print(f'  {name}: {cfg}')
    print()

    trees = read_trees(tree_file) if tree_file is not None else None
    t_max = meta.get('t_max')
    if trees is not None:
        print(f"Training tree from {tree_file} ({len(trees)} tree(s)); branch lengths scaled by 1/height, "
              f"height = {'t_max/2 from the metadata' if t_max is not None else 'root-to-tip path of the tree'}")
        # pickling / unpickling the models walks the linked tree recursively
        sys.setrecursionlimit(max(sys.getrecursionlimit(), 100000))
    tree_info = {}

    models = {}
    for (c1, c2, split_weight), ds in datasets.items():

        print("c1: ", c1, "c2: ", c2, "split_weight: ", split_weight)

        X, y = ds['X'], ds['y']
        train_idx = ds['train_idx']
        W = ds['W'].copy()
        vcv_train = ds['vcv_train']

        X_train = X[train_idx, :].copy()
        y_train = y[train_idx]
        check_response(y_train, response_type)

        X_train_offset = X_train.mean(axis=0)
        X_train_scale = X_train.std(axis=0)

        X_train -= X_train_offset
        X_train /= X_train_scale

        W[train_idx] = 1/len(train_idx)

        # training tree: scaled to the distance matrix and pruned to the training species
        tree_train, spps_train = None, None
        if trees is not None:
            tree_idx = int(ds.get('tree_idx', 0))
            nodes, root, height = scaled_tree(trees[tree_idx], t_max)
            tips = [n.name for n in nodes if is_leaf(n)]
            # row order of X / y: stored by the simulation, else the parser's tip order (as get_D)
            leafs_arr = np.array(ds.get('tip_names', tips))
            if len(leafs_arr) != X.shape[0] or not set(leafs_arr) <= set(tips):
                raise ValueError(f"tree line {tree_idx + 1} of {tree_file} does not match the "
                                 f"{X.shape[0]} rows of the dataset")
            spps_train = leafs_arr[train_idx]
            tree_train = get_training_tree(nodes, spps_train)   # (nodes, root)
            tree_info[(c1, c2, split_weight)] = dict(tree_idx=tree_idx, height=height,
                                                     n_train_tips=sum(is_leaf(n) for n in tree_train[0]))
            print(f"training tree: line {tree_idx + 1}, height {height:.6g} -> 1, "
                  f"{tree_info[(c1, c2, split_weight)]['n_train_tips']} training tips")

        # final and no_agop share the CV (AGOP is only used after CV):
        # if both are requested, no_agop comes out of the final model's fit
        share_cv = 'final' in configs and 'no_agop' in configs

        lkrr_models, best_params = {}, {}
        for name, cfg in configs.items():
            if share_cv and name == 'no_agop':
                continue # already fitted together with final

            print(f'--- variant: {name}')
            lkrr_models[name], best_params[name], lkrr_no_agop = fit_lkrr(
                cfg, X_train, y_train, vcv_train, W[train_idx],
                seed=seed, cores=cores, verbose=verbose,
                folder_dir=folder_dir, name=name, 
                batch_suffix=batch_suffix,
                also_no_agop = share_cv and name == 'final',
                tree_train = tree_train, spps_train = spps_train,
                model_class = model_class, response_type = response_type,
                )
            if lkrr_no_agop is not None:
                lkrr_models['no_agop'] = lkrr_no_agop
                best_params['no_agop'] = best_params[name]
            print()

        lkrr_models = {n: lkrr_models[n] for n in configs} # keep --variants order

        baselines = fit_baselines(response_type, X_train, y_train, seed)

        models[(c1, c2, split_weight)] = {
            'lkrr': lkrr_models,          # {variant_name: fitted Logistic / Multinomial / PhyloKernel}
            'best_params': best_params,   # {variant_name: CV best params}
            **baselines,                  # log_reg + rf (logistic, multinomial) or lin_reg + rf (continuous)
            'X_train_offset': X_train_offset,
            'X_train_scale': X_train_scale,
        }
        print()

    fit_meta = dict(meta, seed=seed, variants=configs,  # keeps sim_uid of the simulation
                    fit_tree_file=tree_file, fit_tree_info=tree_info,
                    response_type=response_type, model_module=model_module,
                    classes=meta.get('classes'))

    out_file = os.path.join(out_dir, f"models_{tag}{batch_suffix}.pkl")
    with open(out_file, 'wb') as f:
        pickle.dump({'meta': fit_meta, 'models': models}, f)

    print(f"Saved {len(models)} fitted model set(s) "
          f"({len(configs)} lkrr variant(s) each) to {out_file}")



def none_or_str(s):
    return None if s.lower() in ('none', 'null', '') else s


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='phyloKQL step 2/3: fit models.')
    # used to locate {folder_dir}/sim_{data_model}_{M1}_{M2}_{ID}.pkl
    parser.add_argument('--data_model', type=str, default='friedman', help='Data model used in the simulation (default: friedman)')
    parser.add_argument('--ID', type=str, default='test', help='Identifier for the experiment (default: test)')
    parser.add_argument('--M1', type=float, default=1.0, help='Parameter M1 (default: 1.0)')
    parser.add_argument('--M2', type=float, default=1.0, help='Parameter M2 (default: 1.0)')
    parser.add_argument('--folder_dir', '-f', type=str, default='out_dir', help='Directory with the simulation and where models are saved (default: out_dir)')
    parser.add_argument('--out_dir', type=str, default='.', help='Directory where models are saved (default: out_dir)')

    parser.add_argument('-p','--process', type=str, default='OU', help="Model process: must match the simulation, or 'none' to ignore the tree whatever the simulation used (default: OU)")
    parser.add_argument('--seed', type=int, default=12038, help='Seed for CV and random forest (default: 12038)')
    parser.add_argument('--cores', type=int, default=5, help='Number of cores (default: 5)')
    parser.add_argument('-v','--verbose', action=argparse.BooleanOptionalAction, default=True, help='Verbose output (default: on; --no-verbose to disable)')
    parser.add_argument('-t', '--tol', type=float, default=1e-6, help='Tolerance for convergence (default: 1e-6)')
    parser.add_argument('-q', '--use_qn', action=argparse.BooleanOptionalAction, default=True, help='Use quasi-Newton method (default: on; --no-use_qn to disable)')
    parser.add_argument('--do_adj', type=none_or_str, choices=[None, 'convex', 'shrink'], default='shrink', help="Adjust latent space (default: shrink; 'none' to disable)")
    parser.add_argument('--do_corr', action=argparse.BooleanOptionalAction, default=True, help='Do correction during CV (default: on; --no-do_corr to disable)')
    parser.add_argument('--corr_scale_error', type=none_or_str, choices=[None, 'auto', 'std', 'n', 'neff', 'std_neff'], default='auto', help="Scale error for correction (default: auto = std for logistic/multinomial, none for continuous; 'none' to disable)")
    parser.add_argument('--do_agop', action=argparse.BooleanOptionalAction, default=True, help='Use average gradient outer product (default: on; --no-do_agop to disable)')
    parser.add_argument('--agop_diag', action=argparse.BooleanOptionalAction, default=True, help='Use diagonal approximation for average gradient outer product (default: on; --no-agop_diag to disable)')
    parser.add_argument('--agop_iter', type=int, default=5, help='Number of iterations for average gradient outer product (default: 5)')
    parser.add_argument('--agop_nogreedy', action='store_true', help='Not use greedy approach for average gradient outer product (default: False)')
    parser.add_argument('--variants', nargs='+', choices=list(VARIANTS), default=list(VARIANTS), help='Model variants to fit, defined relative to the flags above (default: all: %(default)s)')

    parser.add_argument('--batch', type=int, default=None, help='Simulation batch to fit, 0..B-1 (default: none -> unbatched simulation). In a Slurm array: --batch $SLURM_ARRAY_TASK_ID')
    parser.add_argument('--tree_file', type=str, default=None, help='Tree file (one Newick per line, as in the simulation) for CV and inference with the training tree (default: none)')
    parser.add_argument('--response_type', type=str, choices=list(RESPONSE_TYPES), default=None, help='logistic (Logistic), multinomial (Multinomial) or continuous (PhyloKernel) (default: response_type of the .pkl, else logistic)')
    parser.add_argument('--model_module', type=str, default='phyloK.phyloK_v2', help='Module with the model classes, e.g. phyloK_dev2 or phyloK.phyloK_v2 (default: phyloK.phyloK_v2)')
    args = parser.parse_args()

    main(data_model=args.data_model, seed=args.seed, ID=args.ID,
         M1=args.M1, M2=args.M2, process=args.process,
         cores=args.cores, verbose=args.verbose, tol=args.tol,
         use_qn=args.use_qn, folder_dir=args.folder_dir,
         do_adj=args.do_adj, do_corr=args.do_corr,
         corr_scale_error=args.corr_scale_error,
         do_agop=args.do_agop, agop_diag=args.agop_diag,
         agop_iter=args.agop_iter, agop_nogreedy=args.agop_nogreedy,
         variants=args.variants, batch=args.batch,
         out_dir=args.out_dir, tree_file=args.tree_file,
         response_type=args.response_type, model_module=args.model_module,
         )
        

"""
# defaults reproduce the final model, i.e. equivalent to:
python phyloKQL_fit.py -f out_dir --data_model friedman --ID test --M1 1 --M2 1 \
    -p OU --cores 5 --tol 1e-6 -q -v \
    --do_corr --do_adj shrink --corr_scale_error std --do_agop --agop_diag \
    --variants final no_tree no_agop no_corr no_scale

# with the training tree for CV and inference (tree used to build the .pkl):
python phyloKQL_fit_v2.py -f out_dir --data_model friedman --ID test --M1 1 --M2 1 \
    --tree_file test_tree_v5.txt

python phyloKQL_fit_v2.py -f . --data_model avonet\
    --ID Trophic.Level-Carnivore --tree_file tree_avonet_1.0_1.0_Trophic.Level-Carnivore.nwk
"""
# pkl_file = "/Users/ulises/Desktop/MLab/Kernels/Kernel_sims/Kernel_sims/out_dir/models_circle2_1.0_1.0_test.pkl"
# with open(pkl_file, 'rb') as f:
    # models = pickle.load(f)
# np.mean(models['models'][(1.0, 1.0, 5)]['lkrr']['final'].alpha)
# np.mean(models['models'][(1.0, 1.0, 5)]['lkrr']['no_tree'].alpha)
# np.mean(models['models'][(1.0, 1.0, 5)]['lkrr']['no_corr'].alpha)
# np.mean(models['models'][(1.0, 1.0, 5)]['lkrr']['no_scale'].alpha)
# np.mean(models['models'][(1.0, 1.0, 5)]['lkrr']['no_agop'].alpha)
# models['models'][(1.0, 1.0, 5)]['lkrr']['no_agop'].agop_nogreedy