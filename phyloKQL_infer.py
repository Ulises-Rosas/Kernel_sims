"""
Step 3/3 of the phyloKQL pipeline: INFERENCE.

Loads the simulation (phyloKQL_simulate.py) and the fitted models
(phyloKQL_fit.py),

    {sim_dir}/sim_{data_model}_{M1}_{M2}_{ID}[_batch{i}].pkl
    {model_dir}/models_{data_model}_{M1}_{M2}_{ID}[_batch{i}].pkl

(--sim_dir and --model_dir default to --folder_dir/-f)

predicts the test set with every model and writes the summary file in the
same format test_phyloKQL.py produced (rows: true, log_reg, rf, lkrr,
lkrr_corr for the final model, plus lkrr_<variant> and lkrr_<variant>_corr for
every other fitted variant, e.g. lkrr_no_tree, lkrr_no_tree_corr):

    {folder_dir}/summary_results_{data_model}_{M1}_{M2}_{ID}.csv

Batches (simulation run with --B > 1):
  * --batch i : predicts batch i only -> summary_results_..._{ID}_batch{i}.csv
  * no --batch: predicts all B batches (all models_*_batch{i}.pkl must exist)
                and writes one combined summary_results_..._{ID}.csv
"""
import os
import pickle
import argparse
import numpy as np


def get_tag(data_model, M1, M2, ID):
    return f"{data_model}_{M1}_{M2}_{ID}"


def get_batch_suffix(batch):
    return "" if batch is None else f"_batch{batch}"


def predict(sim_file, model_file):
    """Predict the test sets of one simulation file with its fitted models."""

    with open(sim_file, 'rb') as f:
        sim = pickle.load(f)
    datasets = sim['datasets']

    # unpickling needs the model classes (phylokrr_dev) importable
    with open(model_file, 'rb') as f:
        fitted = pickle.load(f)

    if fitted['meta'].get('sim_uid') != sim['meta'].get('sim_uid'):
        raise RuntimeError(f"{model_file} was fitted on a different simulation run "
                           f"than {sim_file} (stale file?); re-run phyloKQL_fit.py")
    models = fitted['models']
    variant_cfgs = fitted['meta']['variants']

    nested_loop_verbose = True

    out = {}
    for key, m in models.items():
        c1, c2, split_weight = key
        ds = datasets[key]
        print("c1: ", c1, "c2: ", c2, "split_weight: ", split_weight)

        X, y = ds['X'], ds['y']
        test_idx = ds['test_idx']

        X_test = X[test_idx, :].copy()
        y_test = y[test_idx]

        X_test -= m['X_train_offset']
        X_test /= m['X_train_scale']

        rows = [('true', y_test)]

        log_reg = m['log_reg']
        rf_model = m['rf']

        rf_loss = np.mean(rf_model.predict(X_test) == y_test)
        std_loss = np.mean(log_reg.predict(X_test) == y_test)

        if nested_loop_verbose:
            print('rf acc: ', rf_loss)
            print('standard log reg acc: ', std_loss)

        rows.append(('log_reg', log_reg.predict_proba(X_test)[:, 1]))
        rows.append(('rf', rf_model.predict_proba(X_test)[:, 1]))

        for name, lkrr in m['lkrr'].items():
            vcv_test_12 = ds['vcv_test_12']
            if variant_cfgs[name]['process'] == 'none':
                # model fitted without tree: R = identity -> zero test/train block
                vcv_test_12 = np.zeros_like(vcv_test_12)

            # final model keeps the original labels (lkrr, lkrr_corr)
            label = 'lkrr' if name == 'final' else f'lkrr_{name}'

            lkrr.do_corr = False
            # just to check how it is doing without correction
            lkrr_probs = lkrr.predict_proba(X_test, vcv_test_12)
            lkrr_loss = np.mean((lkrr_probs >= 0.5) == y_test)

            lkrr.do_corr = True
            lkrr_probs_corr = lkrr.predict_proba(X_test, vcv_test_12)
            lkrr_loss_corr = np.mean((lkrr_probs_corr >= 0.5) == y_test)

            if nested_loop_verbose:
                print(f'[{name}] lkrr (+phylo K) acc: ', lkrr_loss)
                print(f'[{name}] lkrr (+phylo K) acc (corr.): ', lkrr_loss_corr)

            rows.append((label, lkrr_probs))
            rows.append((f'{label}_corr', lkrr_probs_corr))

        out[key] = rows
        print()

    return out


def main(data_model = 'friedman', ID = "test", M1 = 1.0, M2 = 1.0,
         folder_dir = 'out_dir', batch = None,
         sim_dir = None, model_dir = None):

    # simulation and model files can live in different directories;
    # both default to folder_dir, where the summary is written
    sim_dir = folder_dir if sim_dir is None else sim_dir
    model_dir = folder_dir if model_dir is None else model_dir
    dirs = {'sim': sim_dir, 'models': model_dir}

    tag = get_tag(data_model, M1, M2, ID)
    path = lambda kind, b: os.path.join(dirs[kind], f"{kind}_{tag}{get_batch_suffix(b)}.pkl")

    if batch is not None:
        # a single batch
        batches, out_suffix = [batch], get_batch_suffix(batch)

    elif os.path.exists(path('sim', None)):
        # unbatched simulation (--B 1)
        batches, out_suffix = [None], ""

    else:
        # all batches -> one combined summary
        if not os.path.exists(path('sim', 0)):
            raise FileNotFoundError(f"Neither {path('sim', None)} nor {path('sim', 0)} found")
        with open(path('sim', 0), 'rb') as f:
            B = pickle.load(f)['meta']['B']
        batches, out_suffix = list(range(B)), ""

        missing = [path('models', b) for b in batches if not os.path.exists(path('models', b))]
        if missing:
            raise FileNotFoundError(f"{len(missing)} of {B} batch(es) not fitted yet: "
                                    + ", ".join(missing))

    out = {}
    for b in batches:
        if b is not None:
            print(f"=== batch {b}")
        out.update(predict(path('sim', b), path('models', b)))

    # one row per (replicate, model): c1, c2, split_weight, model, values...
    # models: true, log_reg, rf, then lkrr/lkrr_corr (final) and
    # lkrr_<variant>/lkrr_<variant>_corr for every other fitted variant
    os.makedirs(folder_dir, exist_ok=True)
    out_file = os.path.join(folder_dir, f"summary_results_{tag}{out_suffix}.csv")
    with open(out_file, 'w') as f:
        for (c1,c2,split_weight), rows in out.items():
            for label, values in rows:
                f.write(','.join([str(c1), str(c2), str(split_weight), label] + [str(p) for p in values]) + '\n')

    print(f"Saved predictions to {out_file}")


#  'final': PhyloKernel(kernel=rbf, max_iter=100, lam=0.0005092303620343728, warm_start=False, tol=1e-06, center_kernel=False),
#  'no_tree': PhyloKernel(kernel=rbf, max_iter=100, lam=0.0005092303620343728, warm_start=False, tol=1e-06, center_kernel=False),
#  'no_agop': PhyloKernel(kernel=rbf, max_iter=100, lam=0.0005092303620343728, warm_start=False, tol=1e-06, center_kernel=False),
#  'no_corr': PhyloKernel(kernel=rbf, max_iter=100, lam=0.0005092303620343728, warm_start=False, tol=1e-06, center_kernel=False),
#  'no_scale': PhyloKernel(kernel=rbf, max_iter=100, lam=0.0005092303620343728, warm_start=False, tol=1e-06, center_kernel=False)}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='phyloKQL step 3/3: predict the test set.')
    parser.add_argument('--data_model', type=str, default='friedman', help='Data model used in the simulation (default: friedman)')
    parser.add_argument('--ID', type=str, default='test', help='Identifier for the experiment (default: test)')
    parser.add_argument('--M1', type=float, default=1.0, help='Parameter M1 (default: 1.0)')
    parser.add_argument('--M2', type=float, default=1.0, help='Parameter M2 (default: 1.0)')
    parser.add_argument('--folder_dir', '-f', type=str, default='out_dir', help='Output directory for the summary, and default for --sim_dir/--model_dir (default: out_dir)')
    parser.add_argument('--sim_dir', type=str, default=None, help='Directory with the sim_*.pkl files (default: --folder_dir)')
    parser.add_argument('--model_dir', type=str, default=None, help='Directory with the models_*.pkl files (default: --folder_dir)')
    parser.add_argument('--batch', type=int, default=None, help='Predict only this batch, 0..B-1 (default: all batches, combined into one summary)')
    args = parser.parse_args()

    main(data_model=args.data_model, ID=args.ID, M1=args.M1, M2=args.M2,
         folder_dir=args.folder_dir, batch=args.batch,
         sim_dir=args.sim_dir, model_dir=args.model_dir)

"""
python phyloKQL_infer.py -f out_dir --data_model friedman --ID test --M1 1 --M2 1
python phyloKQL_infer.py --sim_dir sims --model_dir models -f results
"""
