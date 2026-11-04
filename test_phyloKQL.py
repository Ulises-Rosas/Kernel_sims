import os
import numpy as np
import argparse


from phyloK.alpha_beta_weighting import (alpha_weighting, 
                                               split_data_idx)
from phyloK.block_cross_validation import k_fold_cv_weigthed
# from phyloK.phyloK_v2 import Logistic
# from phyloK_dev import Logistic
from phyloK_dev2 import Logistic # clip needed
# from phyloK_dev3 import Logistic

from phylokrr.treeio import (parseNewickTree, 
                             is_leaf, get_adj, get_C, 
                             get_APSP, get_BM)

from tree_solve_dev import get_training_tree
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier


def standardize(A, axis=0):
    """Zero mean, unit variance per column (A: n x d or n)."""
    mu = A.mean(axis=axis, keepdims=True)
    sd = A.std(axis=axis, keepdims=True) + 1e-12
    return (A - mu) / sd

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
    
    return D, root, leaves, adj, C, nodes

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


def simulate_data(D, root, leaves, adj, C, 
                  tree,
                  test_num = 0.5, 
                  sigma=1, alpha_ou = 0.5,
                  split_weight = 1/100,
                  sigma_e = 1, alpha_ou_e = 0.5,
                  data_model = 'spiral',
                  bias=None, 
                  process = 'OU',
                  rng = np.random.default_rng(seed=12038),
                  data_process = 'BM',
                  c1 = 1):
    """
    Following binaryPGLMM {ape} code for generating correlated data
    """    
    
    # # Step 1: Define correlation structure for observations
    # get_vcv is a deterministic function, so leaves1 and leaves2 should be the same
    # if process == "OU":

    # V, leaves1 = get_vcv(tree, process=process, sigma2=sigma, alpha=alpha_ou)
    # Ve, leaves2 = get_vcv(tree, process=process, sigma2=sigma_e, alpha=alpha_ou_e)

    names = [n.name for n in leaves]
    # Dsim = D/np.max(D) # scale D to [0,1]
    # sigma is choosen such that the branch lengths are scaled to [0,1] 
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
        D = 2*D/np.max(D)
        V0 = process_D(D, root, leaves, adj, C, 
                       sigma2=1, alpha=1,process='BM')
        R = get_R(V0)  # Get correlation matrix

    elif process == 'none':
        R = np.eye(n)
        
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
        f_ = 1.25 - np.sqrt(X[:, 0]**2 + w * X[:, 1]**2)
        # eta = np.sqrt(c1) * standardize(f_) + b + e        
        eta = f_ + b + e 

    elif data_model == '5-sphere':
        # 5d-sphere
        X = rng.multivariate_normal(np.zeros(n), V, 15).T        
        b = -0.5 if bias is None else bias
        f = 1 - 0.5*np.sqrt(X[:, 0]**2 + X[:, 1]**2 + X[:, 2]**2 + X[:, 3]**2 + X[:, 4]**2)
        # eta = standardize(f)  + b + e
        eta = f  + b + e

    elif data_model == '20-sphere':
        # 20d-sphere
        X = rng.multivariate_normal(np.zeros(n), V, 20).T        
        b = -0.5 if bias is None else bias
        eta = 1 - 0.1*np.sqrt(np.sum(X**2, axis=1)) + b + e

    elif data_model == '20-sphere_':
        # 20d-sphere
        X = rng.multivariate_normal(np.zeros(n), V, 20).T        
        b = -0.5 if bias is None else bias
        f = 1 - 0.1*np.sqrt(np.sum(X[:,:5]**2, axis=1))
        # eta = standardize(f) + b + e
        eta = f + b + e

    elif data_model == 'friedman':
        # Friedman function
        X = rng.multivariate_normal(np.zeros(n), V, 10).T
        # X[:,1] = np.sin(X[:,0])
        # X[:,2] = np.cos(X[:,0])
        # X[:,3] = np.sin(X[:,1]) * np.cos(X[:,2])
        # X[:,4] = 0.25*(X[:,3])**2
        b = -0.5 if bias is None else bias
        f_friedman = 10*np.sin(np.pi*X[:, 0]*X[:, 1]) + 20*(X[:, 2] - 0.5)**2 + 10*X[:, 3] + 5*X[:, 4]
        # eta = np.sqrt(c1) * standardize(f_friedman) + b + e
        eta = f_friedman + b + e

    elif data_model == 'sines':
        X = rng.multivariate_normal(np.zeros(n), V, 10).T
        b = 0.0
        f = 2*np.sin(np.pi *X[:,0]*X[:,1])+ 2*(X[:,2]*X[:,3])**2  +\
            np.sin(2*np.pi*X[:,4]) +\
            0.5*X[:,5]**2
        eta = f + b + e

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

    test_idx, train_idx = split_data_idx(Indx, P, rng)

    return X, y, train_idx, test_idx, R, P


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
         agop_nogreedy = True, # use greedy approach for the average gradient outer product
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
            'lam' : np.logspace(-12, 3, 100, base=2),
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
            'lam' : np.logspace(-12, 3, 100, base=2),
            'gamma'  : np.logspace(-15, 4, 100, base=2),
            'corr' : np.logspace(-10, 0, 100, base=2)
            # 'corr' : np.logspace(-5, 0, 100, base=2)
        }        

    with open(tree_file, 'r') as f: 
        tree = f.read().strip()

    # print(new_tree)

    D, root, leaves, adj, C, nodes = get_D(tree)
    t_max = np.max(D)

    # if t_max exist is available
    height = t_max/2

    # otherwise
    # leaves = [node for node in nodes if is_leaf(node)]
    # height = 0
    # l = leaves[0]
    # while l.ancestor is not None:
    #     height += l.branch_length
    #     l = l.ancestor

    for n in nodes:
        n.branch_length = n.branch_length * (1/height)


    leafs_arr = np.array([n.name for n in leaves])

    
    fix_phi = True
    cv_sample = 100 # number of hyperparameters to test
    cv_folds = 5
    max_iter = 100
    test_num = test_num
    nested_loop_verbose = True


    rng = np.random.default_rng(seed=seed)

    # C1 = 2**rng.uniform(-3, 4, 56)
    # C2 = 2**rng.uniform(-3, 4, 56)

    C1 = np.linspace(1, 2, num = 1)
    C2 = np.linspace(1, 2, num = 1)


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
                lkrr.agop_nogreedy = agop_nogreedy 
                lkrr.agop_diag = agop_diag
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
                            rng = rng,
                            data_process = data_process,
                            c1 = c1
                            )

                if nested_loop_verbose:
                    low_vals = np.sum(W < 1e-8)
                    print(f"Number of probs below 1e-8: {low_vals}")


                X_train, X_test = X[train_idx, :], X[test_idx, :]
                y_train, y_test = y[train_idx], y[test_idx]
                vcv_train = R[train_idx,:][:,train_idx]
                vcv_test_12 = R[test_idx,:][:,train_idx]

                # it is a tuple (nodes, root)
                spps_train = leafs_arr[train_idx]
                tree_train = get_training_tree(nodes, spps_train)
                

                # this keeps time complexity 
                # bellow n^2, but still adaptive enough
                # lkrr.memory_qn = np.sqrt(X_train.shape[0]).astype(int)
                # lkrr.memory_qn = X_train.shape[0]
                lkrr.memory_qn = 20

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
                W[train_idx] = 1/len(train_idx)
                # lkrr.do_agop = do_agop
                import time
                start = time.time()
                # lkrr.tol = 1e-4
                # lkrr.use_bracketing = True
                lkrr_best_params = k_fold_cv_weigthed(
                    X_train, y_train, vcv_train,
                    lkrr,  W[train_idx], params,
                    folds=cv_folds,
                    sample=cv_sample, 
                    verbose= verbose, 
                    seed=seed,
                    cores = cores,
                    nodes = tree_train[0],
                    leaves = spps_train
                )
                # lkrr_best_params = {'lam': np.float64(0.0004584623587215366), 'gamma': np.float64(0.258906851741006), 'corr': np.float64(0.049260178183150566)}
                # lkrr_best_params =   {'lam': np.float64(0.02292311793607683), 'gamma': np.float64(0.06845563584015797), 'corr': np.float64(0.3752329919328113)}
                if nested_loop_verbose:
                    print('lkrr (+ phylo K) CV Time: ', time.time() - start)
                    print('lkrr best params: ', lkrr_best_params)

                lkrr.set_params(**lkrr_best_params)

                if nested_loop_verbose:
                    print('ADDITIVE APPROACH:')

                lkrr.do_agop = do_agop # only applied when we know the hyperparameters.
                lkrr.agop_diag = agop_diag 

                lkrr.set_training_tree(tree_train, leafs_arr[train_idx])
                start = time.time()
                # lkrr.process = 'none'
                lkrr.fit(X_train, y_train, vcv_train)
                print('lkrr fit Time: ', time.time() - start)

                print('agop M:')
                if lkrr.M is not None:
                    print(np.round(lkrr.M, 3))

                    with open(f"{folder_dir}/agop_M.csv", 'w') as f:
                        # write the matrix M to a csv file
                        np.savetxt(f, lkrr.M, delimiter=",") 

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
    parser.add_argument('--agop_iter', type=int, default=5, help='Number of iterations for average gradient outer product (default: 6)')
    parser.add_argument('--agop_nogreedy', action='store_true', help='Not use greedy approach for average gradient outer product (default: False)')

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
          agop_iter=args.agop_iter,
          agop_nogreedy=args.agop_nogreedy,
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
