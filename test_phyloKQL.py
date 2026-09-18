import os
import numpy as np
import argparse


from phylokrr_dev.alpha_beta_weighting import (alpha_weighting, 
                                               split_data_idx)
from phylokrr_dev.block_cross_validation import k_fold_cv_weigthed
# from phylokrr_dev.phyloKPQL import Logistic
from phylokrr_dev.phyloK import Logistic

from phylokrr.treeio import (parseNewickTree, 
                             is_leaf, get_adj, get_C, 
                             get_APSP, get_BM)

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier


def pre_process_V(V):
    n, _ = V.shape
    V = V/np.max(V)
    detV_n = np.linalg.det(V)**(1/n)
    print('detV_n: ', detV_n)
    V = V/detV_n if detV_n != 0 else V
    return V

def get_R( vcv ):
    """
    Get BM correlation matrix from shared path matrix
    """
    kr = 1/np.sqrt(np.diag(vcv))
    R = (kr[:, None] * vcv) * kr
    return R


def get_D(mytree):
    nodes, root = parseNewickTree(mytree)
    assert not isinstance(root.right, list), "Unrooted tree not supported"
    leaves = [n for n in nodes if is_leaf(n)]
    # names = [n.name for n in leaves]

    adj = get_adj(nodes) # O(n)
    C = get_C(nodes, adj) # O(n)

    # O(n^2), it transverses the tree n times, and
    # each traversal is O(n)
    D = get_APSP(leaves, adj, C)
    
    return D, root, leaves, adj, C

def process_D(D, root, leaves, adj, C,
              sigma2, alpha, 
              process = 'BM'):
    
    if process == "BM":
        # makes for each row i it makes n calculations from D.
        # since there n rows, the time complexity is O(n^2)
        BM_D = get_BM(D, root, leaves, adj, C) # O(n^2)
        return sigma2*BM_D
    
    elif process == "OU":
        c1 = 2*alpha
        return (sigma2/c1)*np.exp(-alpha*D)
    
    else:
        raise ValueError("process must be either 'BM' or 'OU'")

def get_from_file(leaves, filename = "./traits_R.csv"):
    
    n = len(leaves)
    from_R = {}
    with open(filename, 'r') as f:
        for i in f.readlines():
            a,b,c = i.split(',')
            if  a == '""':
                continue
            from_R[a.replace('"','')] = [float(b),float(c)]

    k = 0
    X = np.zeros((n, 2))
    for l in leaves:
        X[k,:] = from_R[l]
        k += 1

    return X


def simulate_data(D, root, leaves, adj, C, 
                  tree,
                  test_num = 0.5, 
                  sigma=1, alpha_ou = 0.5,
                  split_weight = 1/100,
                  sigma_e = 1, alpha_ou_e = 0.5,
                  data_model = 'spiral',
                  bias=None, 
                  process = 'OU',
                  spps_test = [],
                  rng = np.random.default_rng(seed=12038),
                  data_process = 'BM'):
    """
    Following binaryPGLMM {ape} code for generating correlated data
    """    
    
    # # Step 1: Define correlation structure for observations
    # get_vcv is a deterministic function, so leaves1 and leaves2 should be the same
    # if process == "OU":

    # V, leaves1 = get_vcv(tree, process=process, sigma2=sigma, alpha=alpha_ou)
    # Ve, leaves2 = get_vcv(tree, process=process, sigma2=sigma_e, alpha=alpha_ou_e)

    names = [n.name for n in leaves]

    V = process_D(D, root, leaves, adj, C, 
                  sigma2=sigma, alpha=alpha_ou,
                    process=data_process)
    
    Ve = process_D(D, root, leaves, adj, C, 
                  sigma2=sigma_e, alpha=alpha_ou_e, 
                  process=data_process)

    V = np.round(V, decimals=5) 
    n = V.shape[0]

    if process == 'OU':
        R = 2*D/np.max(D)

    elif process == 'BM':
        R = get_R(V)  # Get correlation matrix

    else:
        raise ValueError("process must be either 'BM' or 'OU'")

    # R = get_R(V)  # Get correlation matrix
    # print(V[0:5,0:5])
    # print(R[0:5,0:5])

    X = rng.multivariate_normal(np.zeros(n), V, 2).T
    e = rng.multivariate_normal(np.zeros(n), Ve, 1).T.ravel()

    # sine + cosine pattern
    if data_model == 'sine_cosine':
        b = -1.5 if bias is None else bias
        eta = 1.5*np.sin(4*X[:, 0]) + 1.5*np.cos(4*X[:, 1]) + b + e

    elif data_model == 'spiral':
        # Spiral pattern
        b = -0.0 if bias is None else bias
        r = np.sqrt(X[:, 0]**2 + X[:, 1]**2)
        theta = np.arctan2(X[:, 1], X[:, 0])
        eta = np.sin(3 * theta - r) + b + e

    elif data_model == 'checkerboard':
        # Sine wave pattern
        b = -0.0 if bias is None else bias
        eta = np.cos(np.pi * X[:, 0]) * np.cos(np.pi * X[:, 1]) + b + e

    elif data_model == 'xor':
        # XOR pattern
        b = -0.5 if bias is None else bias
        eta = X[:, 0]*X[:, 1] + b + e

    elif data_model == 'circle':
        # circle 1    
        # b = -2.5 if bias is None else bias
        b = -0
        eta = X[:, 0]**2 + X[:, 1]**2 + b + e

    elif data_model == 'circle2':
        # circle 2
        w = 1.5 # change this later.
        b = 0.1 if bias is None else bias
        # print('bias: ', b)
        eta = 1.25 - np.sqrt(X[:, 0]**2 + w * X[:, 1]**2) + b + e

    elif data_model == '5-sphere':
        # 5d-sphere
        X = rng.multivariate_normal(np.zeros(n), V, 10).T        
        b = -0.5 if bias is None else bias
        eta = 1 - 0.5*np.sqrt(X[:, 0]**2 + X[:, 1]**2 + X[:, 2]**2 + X[:, 3]**2 + X[:, 4]**2) + b + e

    elif data_model == '20-sphere':
        # 20d-sphere
        X = rng.multivariate_normal(np.zeros(n), V, 20).T        
        b = -0.5 if bias is None else bias
        eta = 1 - 0.1*np.sqrt(np.sum(X**2, axis=1)) + b + e

    elif data_model == '20-sphere_':
        # 20d-sphere
        X = rng.multivariate_normal(np.zeros(n), V, 20).T        
        b = -0.5 if bias is None else bias
        eta = 1 - 0.1*np.sqrt(np.sum(X**2, axis=1)) + b + e

    elif data_model == 'friedman':
        # Friedman function
        X = rng.multivariate_normal(np.zeros(n), V, 10).T
        b = -0.5 if bias is None else bias
        eta = 10*np.sin(np.pi*X[:, 0]*X[:, 1]) + 20*(X[:, 2] - 0.5)**2 + 10*X[:, 3] + 5*X[:, 4] + b + e

    elif data_model == 'linear':
        # linear pattern
        b = -0.6 if bias is None else bias
        # eta = 1.5*X[:, 0] + 1.5*X[:, 1] + b + e
        eta = 1*rng.normal(0, 1.5, size=X.shape[0]) + 1.5*rng.normal(0, 2.5, size=X.shape[0]) + b + e

    else:
        raise ValueError(f"Unknown model: {data_model}")

    p = 1/(1 + np.exp(-eta))
    # random uniform distribution from 0 to 1
    # y  = rng.binomial(1, p).astype(float)
    y  = (p >= 0.5).astype(float)

    Indx = int(test_num*X.shape[0])
    
    A = alpha_weighting(tree, alpha=split_weight, 
                        tol = 1e-15, branch_length= False)
    # match order of rows using leaves1
    P = np.array([A[leaf] for leaf in names])

    # print(np.sum(P <= 1e-8))
    # print(P)

    if not len(spps_test):
        test_idx, train_idx = split_data_idx(Indx, P, rng)
    else:
        spps_test_set = set(spps_test)
        test_idx = set()
        for i,leaf in enumerate(names):
            if leaf in spps_test_set:
                test_idx.add(i)
        
        train_idx = set(range(len(names))) - test_idx
        test_idx = list(test_idx)
        train_idx = list(train_idx)

        P[test_idx] = 0.0
        P /= np.sum(P)

    # print(P)

    return X, y, train_idx, test_idx, R, P

def conditional_latent(model, f_z, f_n, y, R_12):
    y_c = np.clip(y, model.hessian_tol, 1 - model.hessian_tol)
    # y_c = np.clip(y, 1e-2, 1 - 1e-2)
    sig_inv_y = np.log(y_c/(1 - y_c))
    f_z += R_12 @ model.R_inv @ (sig_inv_y - f_n)
    return f_z

def phylocorrected_logistic(lkrgee, Bphyolm, X_test, X_train, y_train, vcv_test_12):
    eta_z = Bphyolm[0] + Bphyolm[1]*X_test[:,0] + Bphyolm[2]*X_test[:,1]
    
    eta_n = Bphyolm[0] + Bphyolm[1]*X_train[:,0] + Bphyolm[2]*X_train[:,1]
    eta_zn = conditional_latent(lkrgee, eta_z, eta_n, y_train, vcv_test_12)
    return np.exp(eta_zn)/(1+np.exp(eta_zn))
    # return np.exp(eta_z)/(1+np.exp(eta_z))


def main(data_model = 'spiral', seed = 12038, 
         test_num = 0.3, ID = "test", M1 = 1, M2 = 1, 
         tree_file = "test_tree_v5.txt", process = 'OU',
         cores = 1, verbose = True, tol = 1e-6, data_process = 'BM',
         use_qn = False, SW = [1,5,10], folder_dir = '.', 
         do_adj = False, # adjust the level of correction for the latent space,
         do_corr = False, # do correction during cv,
         corr_scale_error = None, # divide error by the sample size. Options: [None, 'std', 'n', 'neff']
         do_agop = True, # use average gradient outer product
         agop_iter = 6, # number of iterations for the average gradient outer product
         agop_diag = False, # use diagonal approximation for the average gradient outer product
         ):

    if not do_corr and do_adj:
        raise ValueError("Cannot adjust latent space without correction during CV")


    # # ref: Kernel methods in computational biology, pp. 62
    if not do_adj:
        # params = {
        #     'lam' : np.logspace(-5, 15, 100, base=2),
        #     'gamma'  : np.logspace(-15, 3, 100, base=2),
        # }
        params = {
            'lam' : np.logspace(-10, 3, 100, base=2),
            'gamma'  : np.logspace(-15, 4, 100, base=2),
        }
    else:
        # params = {
        #     'lam' : np.logspace(-5, 15, 100, base=2),
        #     'gamma'  : np.logspace(-15, 3, 100, base=2),
        #     # 'corr' : np.logspace(-10, 0, 100, base=2)
        #     'corr' : np.logspace(-5, 0, 100, base=2)
        # }
        params = {
            'lam' : np.logspace(-10, 3, 100, base=2),
            'gamma'  : np.logspace(-15, 4, 100, base=2),
            'corr' : np.logspace(-10, 0, 100, base=2)
            # 'corr' : np.logspace(-5, 0, 100, base=2)
        }        

    with open(tree_file, 'r') as f: 
        tree = f.read().strip()

    # tree, test_spps = init_split(tree, test_num)
    test_spps = []

    # print(new_tree)

    D, root, leaves, adj, C = get_D(tree)
    t_max = np.max(D)

    
    fix_phi = True
    cv_sample = 100 # number of hyperparameters to test
    cv_folds = 5
    max_iter = 100
    test_num = test_num
    nested_loop_verbose = True


    rng = np.random.default_rng(seed=seed)

    C1 = 2**rng.uniform(-3, 4, 56)
    C2 = 2**rng.uniform(-3, 4, 56)
    # SW = [1,5,10]

    # C1 = np.linspace(1, 2, num = 1)
    # C2 = np.linspace(1, 2, num = 1)

    # C1 = 2**rng.uniform(-3, 4, 1)    
    # C2 = 2**rng.uniform(-3, 4, 2)
    
    # C1 = [ 5 ]
    # C2 = [ 0.5 ]

    # c1_c2_str = "1.80 0.147"
    # c1_c2_str = "1.32 0.131"
    # c1_c2 = c1_c2_str.strip().split()
    # C1 = [float(c1_c2[0])]
    # C2 = [float(c1_c2[1])]

    # SW = [1]
    # SW = [1]


    out = {} 
    for split_weight in SW:
        for c1 in C1:
            for c2 in C2:

                lkrr = Logistic(get_corr= False, tol=tol, 
                                max_iter=max_iter, kernel='rbf', 
                                verbose=not False, 
                                fix_phi = fix_phi)  
                lkrr.add_extraK = True # backward compatibility with previous versions of phyloKQL
                lkrr.process = process
                lkrr.use_quasi_newton = use_qn
                lkrr.do_corr = do_corr
                lkrr.corr_type = do_adj # ignore if do_corr is False, since it is not possible to adjust the latent space without correction during CV
                lkrr.corr_scale_error = corr_scale_error
                lkrr.agop_iter = agop_iter

                log_reg = LogisticRegression()
                rf_model = RandomForestClassifier(n_estimators=100, random_state=seed)


                alpha_ou = M1 / t_max
                sigma = 2 * alpha_ou * c1

                alpha_ou_e = M2 / t_max
                sigma_e = 2 * alpha_ou_e * c2

                print("c1: ", c1, "c2: ", c2, "split_weight: ", split_weight)
                print("alpha_ou: ", alpha_ou, "alpha_ou_e: ", alpha_ou_e, "sigma: ", sigma, "sigma_e:", sigma_e)


                (X, y, train_idx, 
                test_idx, R, W) = simulate_data(
                            D=D,root=root,leaves=leaves,adj=adj,C=C,
                            tree=tree,
                            test_num=test_num,
                            split_weight=split_weight,
                            alpha_ou = alpha_ou,
                            alpha_ou_e = alpha_ou_e,
                            sigma =sigma,
                            sigma_e = sigma_e,
                            data_model = data_model,
                            process = process,
                            spps_test=test_spps,
                            rng = rng,
                            data_process = data_process
                            )

                if nested_loop_verbose:
                    low_vals = np.sum(W < 1e-8)
                    print(f"Number of probs below 1e-8: {low_vals}")


                X_train, X_test = X[train_idx, :], X[test_idx, :]
                y_train, y_test = y[train_idx], y[test_idx]
                vcv_train = R[train_idx,:][:,train_idx]
                vcv_test_12 = R[test_idx,:][:,train_idx]

                # this keeps time complexity 
                # bellow n^2, but still adaptive enough
                lkrr.memory_qn = np.sqrt(X_train.shape[0]).astype(int)
                # lkrr.memory_qn = X_train.shape[0]

                test_ratio =  np.sum(y_test == 1)/len(y_test)
                train_ratio = np.sum(y_train == 1)/len(y_train)

                if nested_loop_verbose:
                    print()
                    print('label_ratio train:', train_ratio)
                    print('label_ratio test:', test_ratio)

                    print('X_train shape: ', X_train.shape)
                    print('X_test shape: ', X_test.shape)

                if (not (0.2 < test_ratio < 0.8)) or (not (0.2 < train_ratio < 0.8)):
                    continue


                X_train_offset = X_train.mean(axis=0)
                X_train_scale = X_train.std(axis=0)

                X_train -= X_train_offset
                X_train /= X_train_scale

                X_test -= X_train_offset
                X_test /= X_train_scale

                # X_train /= np.linalg.norm(X_train, axis=-1).reshape(-1, 1)
                # X_test /= np.linalg.norm(X_test, axis=-1).reshape(-1, 1)

                # lkrr.do_agop = do_agop
                import time
                start = time.time()
                lkrr_best_params = k_fold_cv_weigthed(
                    X_train, y_train, vcv_train,
                    lkrr,  W[train_idx], params,
                    folds=cv_folds,
                    sample=cv_sample, 
                    verbose= verbose, 
                    seed=seed,
                    cores = cores
                )

                if nested_loop_verbose:
                    print('lkrr (+ phylo K) CV Time: ', time.time() - start)
                    print('lkrr best params: ', lkrr_best_params)

                lkrr.set_params(**lkrr_best_params)

                if nested_loop_verbose:
                    print('ADDITIVE APPROACH:')

                lkrr.do_agop = do_agop # only applied when we know the hyperparameters.
                lkrr.agop_diag = agop_diag 
                lkrr.fit(X_train, y_train, vcv_train)
                print('agop M:')
                print(lkrr.M)

                if nested_loop_verbose:
                    print('alpha param:', np.exp(-lkrr.psc))
                    print('signal (meaningless if process is not OU):',lkrr.psc)

                lkrr.do_corr = False
                # just to check how it is doing without correction 
                lkrr_probs = lkrr.predict_proba(X_test, vcv_test_12)
                lkrr_loss = np.mean((lkrr_probs >= 0.5) == y_test)

                if nested_loop_verbose:
                    # print('lkrr phi:', lkrr.phi )
                    print('lkrr (+phylo K) acc: ', lkrr_loss)

        
                lkrr.do_corr = True
                lkrr_probs_corr = lkrr.predict_proba(X_test, vcv_test_12)
                lkrr_loss_corr = np.mean((lkrr_probs_corr >= 0.5) == y_test)

                if nested_loop_verbose:
                    print('lkrr (+phylo K) acc (corr.): ', lkrr_loss_corr)
            

                log_reg.fit(X_train, y_train)
                rf_model.fit(X_train, y_train)
                rf_loss = np.mean(rf_model.predict(X_test) == y_test)
                std_loss = np.mean(log_reg.predict(X_test) == y_test)

                if nested_loop_verbose:
                    print('rf acc: ', rf_loss)
                    print('standard log reg acc: ', std_loss)

                log_reg_probs = log_reg.predict_proba(X_test)[:, 1]                
                rf_probs = rf_model.predict_proba(X_test)[:, 1]


                out[(c1,c2,split_weight)] = [
                    y_test,
                    log_reg_probs, 
                    rf_probs, 
                    lkrr_probs, 
                    lkrr_probs_corr,
                    ]
                print()


    with open(f"{folder_dir}/summary_results_{data_model}_{M1}_{M2}_{ID}.csv", 'w') as f:

        for (c1,c2,split_weight), results in out.items():
            (y_test, log_reg_probs, rf_probs, 
             lkrr_probs, lkrr_probs_corr, 
            #  lkrgee_probs, lkrgee_probs_corrected
             ) = results

            f.write(','.join([str(c1), str(c2), str(split_weight), 'true'] + [str(p) for p in y_test]) + '\n')
            f.write(','.join([str(c1), str(c2), str(split_weight), 'log_reg'] + [str(p) for p in log_reg_probs]) + '\n')
            f.write(','.join([str(c1), str(c2), str(split_weight), 'rf'] + [str(p) for p in rf_probs]) + '\n')
            f.write(','.join([str(c1), str(c2), str(split_weight), 'lkrr'] + [str(p) for p in lkrr_probs]) + '\n')
            f.write(','.join([str(c1), str(c2), str(split_weight), 'lkrr_corr'] + [str(p) for p in lkrr_probs_corr]) + '\n')



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run phyloKQL simulation.')
    parser.add_argument('--data_model', type=str, default='spiral', help='Data model to use (default: spiral)')
    parser.add_argument('-p','--process', type=str, default='OU', help='Stochastic process for data correlation (default: OU)')
    parser.add_argument('--seed', type=int, default=12038, help='Random seed (default: 12038)')
    parser.add_argument('--cores', type=int, default=1, help='Number of cores (default: 1)')
    parser.add_argument('--test_num', type=float, default=0.3, help='Proportion of test data (default: 0.3)')
    parser.add_argument('--tree_file', type=str, default='test_tree_v5.txt', help='Path to the tree file (default: test_tree_v5.txt)')
    parser.add_argument('--ID', type=str, default='test', help='Identifier for the experiment (default: test)')
    parser.add_argument('--M1', type=float, default=1, help='Parameter M1 (default: 1)')
    parser.add_argument('--M2', type=float, default=1, help='Parameter M2 (default: 1)')
    parser.add_argument('-v','--verbose', action='store_true', help='Enable verbose output')
    parser.add_argument('-t', '--tol', type=float, default=1e-6, help='Tolerance for convergence (default: 1e-6)')
    parser.add_argument('-d', '--data_process', type = str, default = 'BM', help = 'Data process for correlation (default: BM)')
    parser.add_argument('-q', '--use_qn', action='store_true', help='Use quasi-Newton method (default: False)')
    parser.add_argument('--SW', nargs='+', type=float, default=[1,5,10], help='List of split weights (default: [1,5,10])')
    parser.add_argument('--folder_dir', '-f', type=str, default='.', help='Directory to save results (default: current directory)')
    parser.add_argument('--do_adj', type=str, choices=[None, 'convex', 'shrink'], default=None, help='Adjust latent space (default: None)')
    parser.add_argument('--do_corr', action='store_true', help='Do correction during CV (default: False)')
    parser.add_argument('--corr_scale_error', type=str, choices=[None, 'std', 'n', 'neff', 'std_neff'], default=None, help='Scale error for correction (default: None)')
    parser.add_argument('--do_agop', action='store_true', help='Use average gradient outer product (default: False)')
    parser.add_argument('--agop_diag', action='store_true', help='Use diagonal approximation for average gradient outer product (default: False)')
    parser.add_argument('--agop_iter', type=int, default=6, help='Number of iterations for average gradient outer product (default: 6)')

    args = parser.parse_args()

    main(data_model=args.data_model, seed=args.seed,
          test_num=args.test_num, ID=args.ID, M1=args.M1, M2=args.M2, tree_file=args.tree_file,
          process=args.process,
          cores=args.cores, verbose=args.verbose, tol=args.tol,
          data_process=args.data_process,
          use_qn=args.use_qn,
          SW=args.SW,
          folder_dir=args.folder_dir,
          do_adj=args.do_adj,
          do_corr=args.do_corr,
          corr_scale_error=args.corr_scale_error,
          do_agop=args.do_agop,
          agop_diag=args.agop_diag,
          agop_iter=args.agop_iter
        )

"""
time python test_phyloKQL.py --data_model circle2 --test_num 0.5\
        --tree_file test_tree_v5.txt --ID test --M1 1 --M2 1 -p OU --cores 5\
        --data_process BM --use_qn --tol 1e-6

# used at the cluster
python test_phyloKQL.py --data_model circle2\
                --M1 1.0 --M2 1.0 --ID 1kspps_armijo_BM\
                --tree_file test_tree_v7.txt -p 'BM'    
"""
