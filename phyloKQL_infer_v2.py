"""
Step 3/3 of the phyloKQL pipeline: INFERENCE.

Version 2: handles the three response types of phyloKQL_fit_v2.py
(meta['response_type'] of the models file; files without it are logistic):
  * logistic    : same output as phyloKQL_infer.py (probability of class 1)
  * multinomial : one row per (model, class), labelled "<model>|<class>", with the
                  class probabilities; "true|<class>" holds the one-hot truth.
                  log_reg / rf columns are aligned to the classes of y (a class
                  absent from their training set gets probability 0)
  * continuous  : rows true, lin_reg, rf, lkrr, lkrr_corr, ... with predicted values
"""
import os
import pickle
import argparse
import numpy as np


def get_tag(data_model, M1, M2, ID):
    return f"{data_model}_{M1}_{M2}_{ID}"


def get_batch_suffix(batch):
    return "" if batch is None else f"_batch{batch}"


def class_names(meta, m):
    names = meta.get('classes')
    return list(names) if names is not None and len(names) == m else [str(k) for k in range(m)]


def proba_all_classes(clf, X_test, m):
    """sklearn predict_proba with one column per class 0..m-1 (classes absent in training -> 0)"""
    P = np.zeros((X_test.shape[0], m))
    P[:, clf.classes_.astype(int)] = clf.predict_proba(X_test)
    return P


def vcv_for(name, ds, variant_cfgs):
    """test/train correlation block; zero for a model fitted without tree"""
    vcv_test_12 = ds['vcv_test_12']
    if variant_cfgs[name]['process'] == 'none':
        # model fitted without tree: R = identity -> zero test/train block
        vcv_test_12 = np.zeros_like(vcv_test_12)
    return vcv_test_12


def predict_logistic(m, ds, X_test, y_test, variant_cfgs, nested_loop_verbose):
    """same rows as phyloKQL_infer.py"""
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
    return rows


def predict_multinomial(m, ds, X_test, y_test, variant_cfgs, nested_loop_verbose, classes):
    """rows '<model>|<class>' with class probabilities; 'true|<class>' one-hot"""
    n_cls = y_test.shape[1]
    y_lab = np.argmax(y_test, axis=1)
    rows = []
    add = lambda label, P: rows.extend((f"{label}|{c}", P[:, k]) for k, c in enumerate(classes))

    add('true', y_test)
    for name in ('log_reg', 'rf'):
        P = proba_all_classes(m[name], X_test, n_cls)
        if nested_loop_verbose:
            print(f'{name} acc: ', np.mean(np.argmax(P, axis=1) == y_lab))
        add(name, P)

    for name, lkrr in m['lkrr'].items():
        vcv_test_12 = vcv_for(name, ds, variant_cfgs)
        label = 'lkrr' if name == 'final' else f'lkrr_{name}'

        lkrr.do_corr = False
        P = lkrr.predict_proba(X_test, vcv_test_12)
        lkrr.do_corr = True
        P_corr = lkrr.predict_proba(X_test, vcv_test_12)

        if nested_loop_verbose:
            print(f'[{name}] lkrr (+phylo K) acc: ', np.mean(np.argmax(P, axis=1) == y_lab))
            print(f'[{name}] lkrr (+phylo K) acc (corr.): ', np.mean(np.argmax(P_corr, axis=1) == y_lab))
        add(label, P)
        add(f'{label}_corr', P_corr)
    return rows


def predict_continuous(m, ds, X_test, y_test, variant_cfgs, nested_loop_verbose):
    """rows true, lin_reg, rf, lkrr, lkrr_corr, ... with predicted values"""
    rmse = lambda p: np.sqrt(np.mean((p - y_test)**2))
    rows = [('true', y_test)]
    for name in ('lin_reg', 'rf'):
        pred = m[name].predict(X_test)
        if nested_loop_verbose:
            print(f'{name} rmse: ', rmse(pred))
        rows.append((name, pred))

    for name, lkrr in m['lkrr'].items():
        vcv_test_12 = vcv_for(name, ds, variant_cfgs)
        label = 'lkrr' if name == 'final' else f'lkrr_{name}'

        lkrr.do_corr = False
        pred = lkrr.predict(X_test, vcv_test_12)
        lkrr.do_corr = True
        pred_corr = lkrr.predict(X_test, vcv_test_12)

        if nested_loop_verbose:
            print(f'[{name}] lkrr (+phylo K) rmse: ', rmse(pred))
            print(f'[{name}] lkrr (+phylo K) rmse (corr.): ', rmse(pred_corr))
        rows.append((label, pred))
        rows.append((f'{label}_corr', pred_corr))
    return rows


def predict(sim_file, model_file):
    """Predict the test sets of one simulation file with its fitted models."""

    with open(sim_file, 'rb') as f:
        sim = pickle.load(f)
    datasets = sim['datasets']

    # unpickling needs the model classes (newname) importable
    with open(model_file, 'rb') as f:
        fitted = pickle.load(f)

    if fitted['meta'].get('sim_uid') != sim['meta'].get('sim_uid'):
        raise RuntimeError(f"{model_file} was fitted on a different simulation run "
                           f"than {sim_file} (stale file?); re-run phyloKQL_fit.py")
    models = fitted['models']
    variant_cfgs = fitted['meta']['variants']
    response_type = fitted['meta'].get('response_type', 'logistic')
    print(f"response type: {response_type}")

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

        if response_type == 'logistic':
            rows = predict_logistic(m, ds, X_test, y_test, variant_cfgs, nested_loop_verbose)
        elif response_type == 'multinomial':
            rows = predict_multinomial(m, ds, X_test, y_test, variant_cfgs, nested_loop_verbose,
                                       class_names(fitted['meta'], y_test.shape[1]))
        elif response_type == 'continuous':
            rows = predict_continuous(m, ds, X_test, y_test, variant_cfgs, nested_loop_verbose)
        else:
            raise ValueError(f"Unknown response type {response_type}")

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
    parser = argparse.ArgumentParser(description='phyloKQL step 3/3 (v2: logistic, multinomial, continuous): predict the test set.')
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
python phyloKQL_infer_v2.py -f out_dir --data_model friedman --ID test --M1 1 --M2 1
python phyloKQL_infer_v2.py --sim_dir sims --model_dir models -f results
"""
