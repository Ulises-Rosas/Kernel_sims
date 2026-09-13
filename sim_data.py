import os
import numpy as np
import argparse


from phylokrr_dev.alpha_beta_weighting import (alpha_weighting, 
                                               split_data_idx)

from phylokrr.treeio import (parseNewickTree, 
                             is_leaf, get_adj, get_C, 
                             get_APSP, get_BM)

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


    test_idx, train_idx = split_data_idx(Indx, P, rng)
    return X, y, train_idx, test_idx, R, P


def main(data_model = 'spiral', seed = 12038, 
         test_num = 0.3, ID = "test", M1 = 1, M2 = 1, 
         tree_file = "test_tree_v5.txt", process = 'OU', data_process = 'BM',
         SW = [1,5,10], folder_dir = '.', ):


    with open(tree_file, 'r') as f: 
        tree = f.read().strip()

    D, root, leaves, adj, C = get_D(tree)
    t_max = np.max(D)

    
    test_num = test_num
    rng = np.random.default_rng(seed=seed)

    # C1 = 2**rng.uniform(-3, 4, 56)
    # C2 = 2**rng.uniform(-3, 4, 56)


    C1 = np.linspace(1, 2, num = 1)
    C2 = np.linspace(1, 2, num = 1)

    all_params = []
    k = 0
    for split_weight in SW:
        for c1 in C1:
            for c2 in C2:
                all_params.append( (split_weight, c1, c2, k) )
                k += 1

    all_params = np.array(all_params)
    np.save(f"{folder_dir}/all_params_{ID}.npy", all_params)

    # print("all_params: ", all_params)
    for  (split_weight, c1, c2, i) in all_params:

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
                    data_process = data_process
                    )

        y_train, y_test = y[train_idx], y[test_idx]
        test_ratio =  np.sum(y_test == 1)/len(y_test)
        train_ratio = np.sum(y_train == 1)/len(y_train)

        print('label_ratio train:', train_ratio)
        print('label_ratio test:', test_ratio)


        if (not (0.2 < test_ratio < 0.8)) or (not (0.2 < train_ratio < 0.8)):
            continue

        np.save(f"{folder_dir}/X_{ID}_{i}.npy", X)
        np.save(f"{folder_dir}/y_{ID}_{i}.npy", y)
        np.save(f"{folder_dir}/train_idx_{ID}_{i}.npy", train_idx)
        np.save(f"{folder_dir}/test_idx_{ID}_{i}.npy", test_idx)
        np.save(f"{folder_dir}/R_{ID}_{i}.npy", R)
        np.save(f"{folder_dir}/W_{ID}_{i}.npy", W)
        # save leaves names
        with open(f"{folder_dir}/leaves_{ID}_{i}.txt", 'w') as f:
            for leaf in leaves:
                f.write(f"{leaf.name}\n")

    


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run data simulation.')
    parser.add_argument('--data_model', type=str, default='spiral', help='Data model to use (default: spiral)')
    parser.add_argument('--seed', type=int, default=12038, help='Random seed (default: 12038)')
    parser.add_argument('--test_num', type=float, default=0.3, help='Proportion of test data (default: 0.3)')
    parser.add_argument('--tree_file', type=str, default='test_tree_v5.txt', help='Path to the tree file (default: test_tree_v5.txt)')
    parser.add_argument('--ID', type=str, default='test', help='Identifier for the experiment (default: test)')
    parser.add_argument('--M1', type=float, default=1, help='Parameter M1 (default: 1)')
    parser.add_argument('--M2', type=float, default=1, help='Parameter M2 (default: 1)')
    parser.add_argument('-p','--process', type=str, default='OU', help='Process assumed in the model (default: OU)')
    parser.add_argument('-d', '--data_process', type = str, default = 'BM', help = 'Process for data generation (default: BM)')
    parser.add_argument('--SW', nargs='+', type=float, default=[1,5,10], help='List of split weights (default: [1,5,10])')
    parser.add_argument('--folder_dir', '-f', type=str, default='.', help='Directory to save results (default: current directory)')
    args = parser.parse_args()



    main(data_model = args.data_model, seed = args.seed, 
         test_num = args.test_num, ID = args.ID, M1 = args.M1, M2 = args.M2, 
         tree_file = args.tree_file, process = args.process, data_process = args.data_process,
         SW = args.SW, folder_dir = args.folder_dir)

"""
python sim_data.py --data_model circle2 --seed 12038 --test_num 0.5\
    --tree_file test_tree_v7.txt --ID circle_data --M1 1 --M2 1\
    --process OU --data_process BM --SW 1 --folder_dir out_dir
"""


# region: old code
                # lkrgee = Logistic(get_corr = False, tol = tol, 
                #                 max_iter = max_iter, kernel = 'rbf', 
                #                 verbose = True,
                #                 fix_phi = fix_phi)
                # lkrgee.process = process

                # import time
                # start = time.time()
                # lkrgee_best_params = k_fold_cv_weigthed(
                #     X_train, y_train, vcv_train,
                #     lkrgee,  W[train_idx], params,
                #     folds=cv_folds,
                #     sample=cv_sample, 
                #     verbose=not False, 
                #     seed=seed
                # )
                # # print(W[train_idx])
                # if nested_loop_verbose:
                #     print('lkrgee CV Time: ', time.time() - start)
                #     print('lkrgee best params: ', lkrgee_best_params)

                # lkrgee.set_params(**lkrgee_best_params)
                # lkrgee.fit(X_train, y_train, vcv_train)

                # lkrgee_loss = np.mean(lkrgee.predict(X_test, vcv_test_12) == y_test)
                # lkrgee_loss_nc = np.mean(lkrgee.predict(X_test, None) == y_test)

                # if nested_loop_verbose:
                #     print('alpha param:', np.exp(-lkrgee.psc))
                #     print('signal (meaningless if process is not OU):',lkrgee.psc)

                    # print('lkrgee acc: ', lkrgee_loss_nc)
                    # print('lkrgee acc (corr.): ', lkrgee_loss)
# endregion: old code