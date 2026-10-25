"""
Step 2/3 of the phyloKQL pipeline: FITTING.

Loads the simulated data written by phyloKQL_simulate.py,

    {folder_dir}/sim_{data_model}_{M1}_{M2}_{ID}.pkl          (or ..._batch{i}.pkl with --batch i)

and, for every simulated replicate, standardizes the features and, for each
model variant in VARIANTS (final model = CLI flags; no_tree = -p none;
no_agop; no_corr; no_scale), runs the weighted k-fold CV for the Logistic
(phylo) kernel model and refits it with the best hyperparameters (+ AGOP if
requested). It also fits the baselines (logistic regression and random
forest). Everything is saved to

    {folder_dir}/models_{data_model}_{M1}_{M2}_{ID}.pkl       (or ..._batch{i}.pkl with --batch i)

which is consumed by phyloKQL_infer.py.
"""
import os
import copy
import time
import pickle
import argparse
import numpy as np

from phylokrr_dev.block_cross_validation import k_fold_cv_weigthed
from phylokrr_dev.phyloK_v2 import Logistic
# from phyloK_dev import Logistic
# from phyloK_dev2 import Logistic

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier


def get_tag(data_model, M1, M2, ID):
    return f"{data_model}_{M1}_{M2}_{ID}"


def get_batch_suffix(batch):
    return "" if batch is None else f"_batch{batch}"

# Model variants fitted on every simulated replicate. Each one is defined as a
# change with respect to the "final" model, i.e. the configuration given by
# the CLI flags (whose defaults are the final model).
VARIANTS = {
    'final'   : {},                                        # -p OU + corr + adj + scale + AGOP
    'no_tree' : {'process': 'none', 'do_corr': False, 'do_adj': None},                       # -p none (R = identity)
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


def fit_lkrr(cfg, X_train, y_train, vcv_train, W_train,
             seed, cores, verbose, folder_dir, name, batch_suffix = "",
             also_no_agop = False):
    """CV + refit of one Logistic (phylo) kernel model with configuration cfg.

    If also_no_agop, the model is first fitted without AGOP (that copy is
    returned as the no_agop model) and then with AGOP, sharing the same CV.
    """

    fix_phi = True
    cv_sample = 100 # number of hyperparameters to test
    cv_folds = 5
    max_iter = 100
    nested_loop_verbose = True

    if cfg['process'] == 'none':
        # ignore the tree regardless of how the data were simulated
        vcv_train = np.eye(vcv_train.shape[0])

    lkrr = Logistic(get_corr= False, tol=cfg['tol'],
                    max_iter=max_iter, kernel='rbf',
                    verbose=not False,
                    fix_phi = fix_phi)
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

    start = time.time()
    lkrr_best_params = k_fold_cv_weigthed(
        X_train, y_train, vcv_train,
        lkrr,  W_train, get_params(cfg['do_adj']),
        folds=cv_folds,
        sample=cv_sample,
        verbose= verbose,
        seed=seed,
        cores = cores
    )
    if nested_loop_verbose:
        print(f'[{name}] lkrr (+ phylo K) CV Time: ', time.time() - start)
        print(f'[{name}] lkrr best params: ', lkrr_best_params)

    lkrr.set_params(**lkrr_best_params)

    if nested_loop_verbose:
        print(f'[{name}] ADDITIVE APPROACH:')

    lkrr_no_agop = None
    if also_no_agop:
        # 1) fit without AGOP and keep a copy of that model
        lkrr.do_agop = False
        lkrr.agop_diag = False
        lkrr.fit(X_train, y_train, vcv_train)
        lkrr_no_agop = copy.deepcopy(lkrr)

    # 2) fit with this variant's AGOP settings
    lkrr.do_agop = cfg['do_agop'] # only applied when we know the hyperparameters.
    lkrr.agop_diag = cfg['agop_diag']
    start = time.time()
    lkrr.fit(X_train, y_train, vcv_train)

    if nested_loop_verbose:
        print(f'[{name}] lkrr (+ phylo K) fit Time: ', time.time() - start)

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
         corr_scale_error = 'std', # divide error by the sample size. Options: [None, 'std', 'n', 'neff']
         do_agop = True, # use average gradient outer product
         agop_iter = 5, # number of iterations for the average gradient outer product
         agop_diag = True, # use diagonal approximation for the average gradient outer product
         agop_nogreedy = False, # use greedy approach for the average gradient outer product
         variants = tuple(VARIANTS), # which model variants to fit (see VARIANTS)
         batch = None, # simulation batch to fit (None -> unbatched simulation, --B 1)
         out_dir = 'out_dir'
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

    print('Model variants to fit:')
    for name, cfg in configs.items():
        print(f'  {name}: {cfg}')
    print()

    models = {}
    for (c1, c2, split_weight), ds in datasets.items():

        print("c1: ", c1, "c2: ", c2, "split_weight: ", split_weight)

        X, y = ds['X'], ds['y']
        train_idx = ds['train_idx']
        W = ds['W'].copy()
        vcv_train = ds['vcv_train']

        X_train = X[train_idx, :].copy()
        y_train = y[train_idx]

        X_train_offset = X_train.mean(axis=0)
        X_train_scale = X_train.std(axis=0)

        X_train -= X_train_offset
        X_train /= X_train_scale

        W[train_idx] = 1/len(train_idx)

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
                also_no_agop = share_cv and name == 'final'
                )
            if lkrr_no_agop is not None:
                lkrr_models['no_agop'] = lkrr_no_agop
                best_params['no_agop'] = best_params[name]
            print()

        lkrr_models = {n: lkrr_models[n] for n in configs} # keep --variants order

        log_reg = LogisticRegression()
        rf_model = RandomForestClassifier(n_estimators=100, random_state=seed)
        log_reg.fit(X_train, y_train)
        rf_model.fit(X_train, y_train)

        models[(c1, c2, split_weight)] = {
            'lkrr': lkrr_models,          # {variant_name: fitted Logistic}
            'best_params': best_params,   # {variant_name: CV best params}
            'log_reg': log_reg,
            'rf': rf_model,
            'X_train_offset': X_train_offset,
            'X_train_scale': X_train_scale,
        }
        print()

    fit_meta = dict(meta, seed=seed, variants=configs)  # keeps sim_uid of the simulation

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
    parser.add_argument('--corr_scale_error', type=none_or_str, choices=[None, 'std', 'n', 'neff', 'std_neff'], default='std', help="Scale error for correction (default: std; 'none' to disable)")
    parser.add_argument('--do_agop', action=argparse.BooleanOptionalAction, default=True, help='Use average gradient outer product (default: on; --no-do_agop to disable)')
    parser.add_argument('--agop_diag', action=argparse.BooleanOptionalAction, default=True, help='Use diagonal approximation for average gradient outer product (default: on; --no-agop_diag to disable)')
    parser.add_argument('--agop_iter', type=int, default=5, help='Number of iterations for average gradient outer product (default: 5)')
    parser.add_argument('--agop_nogreedy', action='store_true', help='Not use greedy approach for average gradient outer product (default: False)')
    parser.add_argument('--variants', nargs='+', choices=list(VARIANTS), default=list(VARIANTS), help='Model variants to fit, defined relative to the flags above (default: all: %(default)s)')

    parser.add_argument('--batch', type=int, default=None, help='Simulation batch to fit, 0..B-1 (default: none -> unbatched simulation). In a Slurm array: --batch $SLURM_ARRAY_TASK_ID')
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
         out_dir=args.out_dir
         )
        

"""
# defaults reproduce the final model, i.e. equivalent to:
python phyloKQL_fit.py -f out_dir --data_model friedman --ID test --M1 1 --M2 1 \
    -p OU --cores 5 --tol 1e-6 -q -v \
    --do_corr --do_adj shrink --corr_scale_error std --do_agop --agop_diag \
    --variants final no_tree no_agop no_corr no_scale
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