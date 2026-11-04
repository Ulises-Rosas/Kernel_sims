"""
Step 1/3 of the phyloKQL pipeline: SIMULATION.

Simulates the data (X, y), the phylogenetically-weighted train/test split and
the tree correlation matrix R used by the model, exactly as test_phyloKQL.py
did, for every combination of split weight (SW) x c1 x c2, and saves
everything to

    {folder_dir}/sim_{data_model}_{M1}_{M2}_{ID}.pkl                 (--B 1)
    {folder_dir}/sim_{data_model}_{M1}_{M2}_{ID}_batch{i}.pkl, i = 0..B-1  (--B > 1)

which are consumed by phyloKQL_fit.py and phyloKQL_infer.py (use --batch i
there to fit/predict one batch, e.g. one Slurm array task per batch).

c1 and c2 (variance scalings of the data and error processes):
  * default (no --N): c1 = c2 = 1, as in test_phyloKQL.py
  * --N N: C1 = 2**rng.uniform(lb1, ub1, N) and C2 = 2**rng.uniform(lb2, ub2, N)
    with (lb1, ub1) = --C1_range and (lb2, ub2) = --C2_range. All N x N
    (c1, c2) pairs are simulated for each split weight.

Trees (--tree_file, one Newick tree per line; blank lines and lines starting
with '#' are ignored). With T trees and S = len(SW) x len(C1) x len(C2)
target simulations:
  * T == 1: that tree is used for every simulation
  * T <  S: S trees are sampled at random WITH replacement
  * T == S: tree i is used for simulation i (in loop order SW > c1 > c2)
  * T >  S: S trees are sampled at random WITHOUT replacement
Each dataset stores which tree it came from: 'tree_line' (1-based line
number in --tree_file) and 'tree_idx' (0-based index among the kept trees),
plus 'tip_names', the tip label of each row of X/y (used by phyloglm_fit.py);
the run metadata stores the same for every target simulation.

All datasets are simulated in a single pass with one RNG and only then split
into batches, so the simulated data do not depend on --B.

Note: the RNG draws do not depend on -p/--process, so for a fixed seed the
simulated X, y and splits are identical across processes; only R changes.
"""
import os
import glob
import functools
import uuid
import pickle
import argparse
import numpy as np


from phyloK.alpha_beta_weighting import (alpha_weighting, 
                                               split_data_idx)

from phylokrr.treeio import (parseNewickTree, 
                             is_leaf, get_adj, get_C, 
                             get_APSP, get_BM)


def get_tag(data_model, M1, M2, ID):
    return f"{data_model}_{M1}_{M2}_{ID}"


def get_batch_suffix(batch):
    return "" if batch is None else f"_batch{batch}"


def read_trees(tree_file):
    """One Newick tree per line; skips blank lines and '#' comments.

    Returns the trees and their 1-based line numbers in tree_file.
    """
    trees, lines = [], []
    with open(tree_file, 'r') as f:
        for line_no, l in enumerate(f, start=1):
            t = l.strip()
            if t and not t.startswith('#'):
                trees.append(t)
                lines.append(line_no)
    if not trees:
        raise ValueError(f"No trees found in {tree_file}")
    return trees, lines


def assign_trees(n_trees, n_sims, rng):
    """Tree index for each of the n_sims simulations (see module docstring)."""
    if n_trees == 1:
        return np.zeros(n_sims, dtype=int)
    if n_trees == n_sims:
        return np.arange(n_sims)
    # fewer trees than simulations -> with replacement; more -> without
    return rng.choice(n_trees, size=n_sims, replace=n_trees < n_sims)


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
    # print("data_process: ", data_process)
    # print('D[0:5,0:5]: ', D[0:5,0:5])
    # print("V[0:5,0:5]: ", V[0:5,0:5])
    # print("Ve[0:5,0:5]: ", Ve[0:5,0:5])

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


def main(data_model = 'friedman', seed = 12038, 
         test_num = 0.5, ID = "test", M1 = 1.0, M2 = 1.0, 
         tree_file = "test_tree_v5.txt", process = 'OU',
         data_process = 'BM', SW = [5.0], folder_dir = 'out_dir',
         verbose = True,
         N = None, # number of c1 and c2 values; None -> c1 = c2 = 1
         C1_range = (-3, 4), # log2 range of c1 when N is given
         C2_range = (-3, 4), # log2 range of c2 when N is given
         B = 1, # number of batch files
         ):

    if B < 1:
        raise ValueError("--B must be >= 1")

    os.makedirs(folder_dir, exist_ok=True)

    trees, tree_lines = read_trees(tree_file)

    # D (all pairwise path lengths) is O(n^2) per tree: compute it once per
    # tree and keep the most recently used ones in memory
    @functools.lru_cache(maxsize=16)
    def tree_info(tree_idx):
        return get_D(trees[tree_idx])

    rng = np.random.default_rng(seed=seed)

    if N is None:
        # final model: c1 = c2 = 1 (no RNG draws, so data match test_phyloKQL.py)
        C1 = np.linspace(1, 2, num = 1)
        C2 = np.linspace(1, 2, num = 1)
    else:
        C1 = 2**rng.uniform(C1_range[0], C1_range[1], N)
        C2 = 2**rng.uniform(C2_range[0], C2_range[1], N)

    print("C1: ", C1)
    print("C2: ", C2)
    n_sims = len(SW)*len(C1)*len(C2)
    print(f"Simulating {len(SW)} x {len(C1)} x {len(C2)} = "
          f"{n_sims} dataset(s)")

    # with a single tree (or one tree per simulation) no RNG draws are made
    tree_idxs = assign_trees(len(trees), n_sims, rng)
    print(f"Trees in {tree_file}: {len(trees)} -> ", end='')
    if len(trees) == 1:
        print("same tree for every simulation")
    elif len(trees) == n_sims:
        print("one tree per simulation")
    elif len(trees) < n_sims:
        print(f"sampled with replacement ({len(np.unique(tree_idxs))} distinct)")
    else:
        print("sampled without replacement")
    print()

    datasets = {}
    sim_i = -1
    n_tips = None
    for split_weight in SW:
        for c1 in C1:
            for c2 in C2:
                sim_i += 1
                tree_idx = int(tree_idxs[sim_i])
                tree = trees[tree_idx]
                D, root, leaves, adj, C = tree_info(tree_idx)
                t_max = np.max(D)

                # all trees are assumed to have the same number of tips
                if n_tips is None:
                    n_tips = len(leaves)
                elif len(leaves) != n_tips:
                    print(f"Warning: tree on line {tree_lines[tree_idx]} has "
                          f"{len(leaves)} tips, expected {n_tips}")

                alpha_ou = M1 / t_max
                sigma = 2 * alpha_ou * c1

                alpha_ou_e = M2 / t_max
                sigma_e = 2 * alpha_ou_e * c2

                print("c1: ", c1, "c2: ", c2, "split_weight: ", split_weight,
                      "tree line: ", tree_lines[tree_idx], f"({len(leaves)} tips)")
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

                if verbose:
                    low_vals = np.sum(W < 1e-8)
                    print(f"Number of probs below 1e-8: {low_vals}")

                y_train, y_test = y[train_idx], y[test_idx]
                test_ratio =  np.sum(y_test == 1)/len(y_test)
                train_ratio = np.sum(y_train == 1)/len(y_train)

                if verbose:
                    print()
                    print('label_ratio train:', train_ratio)
                    print('label_ratio test:', test_ratio)
                    print('X_train shape: ', X[train_idx, :].shape)
                    print('X_test shape: ', X[test_idx, :].shape)

                # same filter as in test_phyloKQL.py: unbalanced
                # replicates are not fitted
                if (not (0.2 < test_ratio < 0.8)) or (not (0.2 < train_ratio < 0.8)):
                    print('Skipping (unbalanced labels)')
                    print()
                    continue

                datasets[(c1, c2, split_weight)] = {
                    'X': X,
                    'y': y,
                    'train_idx': train_idx,
                    'test_idx': test_idx,
                    'W': W,
                    'vcv_train': R[train_idx,:][:,train_idx],
                    'vcv_test_12': R[test_idx,:][:,train_idx],
                    'alpha_ou': alpha_ou, 'sigma': sigma,
                    'alpha_ou_e': alpha_ou_e, 'sigma_e': sigma_e,
                    'tree_idx': tree_idx,
                    'tree_line': tree_lines[tree_idx],
                    'tip_names': [n.name for n in leaves], # row order of X, y and R
                }
                print()

    meta = dict(data_model=data_model, seed=seed, test_num=test_num,
                ID=ID, M1=M1, M2=M2, tree_file=tree_file,
                process=process, data_process=data_process, SW=list(SW),
                N=N, C1_range=list(C1_range), C2_range=list(C2_range),
                C1=list(C1), C2=list(C2), B=B,
                n_trees=len(trees), tree_idxs=[int(t) for t in tree_idxs],
                tree_lines=[tree_lines[int(t)] for t in tree_idxs], # per target simulation, loop order
                t_max=float(t_max),
                sim_uid=uuid.uuid4().hex) # lets fit/infer detect stale files

    tag = get_tag(data_model, M1, M2, ID)

    # remove simulation files left by a previous run of this tag with a
    # different --B, so fit/infer never pick up a stale layout
    for old in ([os.path.join(folder_dir, f"sim_{tag}.pkl")] +
                glob.glob(os.path.join(folder_dir, f"sim_{tag}_batch*.pkl"))):
        if os.path.exists(old):
            os.remove(old)
    keys = list(datasets.keys())

    if B > len(keys):
        print(f"Warning: --B {B} > number of kept datasets ({len(keys)}); "
              f"some batch files will be empty")

    # contiguous, balanced chunks (sizes differ by at most one)
    chunks = np.array_split(np.arange(len(keys)), B)
    for b, idx in enumerate(chunks):
        batch = None if B == 1 else b
        batch_keys = [keys[i] for i in idx]

        out_file = os.path.join(folder_dir, f"sim_{tag}{get_batch_suffix(batch)}.pkl")
        with open(out_file, 'wb') as f:
            pickle.dump({'meta': dict(meta, batch=batch),
                         'datasets': {k: datasets[k] for k in batch_keys}}, f)

        print(f"Saved {len(batch_keys)} simulated dataset(s) to {out_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='phyloKQL step 1/3: simulate data.')
    parser.add_argument('--data_model', type=str, default='friedman', help='Data model to use (default: friedman)')
    parser.add_argument('-p','--process', type=str, default='OU', help='Process for the tree correlation matrix R used by the model: OU, BM or none (default: OU)')
    parser.add_argument('--seed', type=int, default=12038, help='Random seed (default: 12038)')
    parser.add_argument('--test_num', type=float, default=0.5, help='Proportion of test data (default: 0.5)')
    parser.add_argument('--tree_file', type=str, default='test_tree_v5.txt', help='Tree file, one Newick tree per line: 1 tree -> used for all simulations; fewer than the simulations -> sampled with replacement; as many -> one each; more -> sampled without replacement (default: test_tree_v5.txt)')
    parser.add_argument('--ID', type=str, default='test', help='Identifier for the experiment (default: test)')
    parser.add_argument('--M1', type=float, default=1.0, help='Parameter M1 (default: 1.0)')
    parser.add_argument('--M2', type=float, default=1.0, help='Parameter M2 (default: 1.0)')
    parser.add_argument('-d', '--data_process', type = str, default = 'BM', help = 'Data process for correlation (default: BM)')
    parser.add_argument('--SW', nargs='+', type=float, default=[5.0], help='List of split weights (default: [5.0])')
    parser.add_argument('--folder_dir', '-f', type=str, default='out_dir', help='Directory to save results (default: out_dir)')
    parser.add_argument('-v','--verbose', action=argparse.BooleanOptionalAction, default=True, help='Verbose output (default: on; disable with --no-verbose)')
    parser.add_argument('--N', type=int, default=None, help='Number of c1 and c2 values drawn as 2**uniform(lb, ub, N); all N x N pairs are simulated (default: none -> c1 = c2 = 1)')
    parser.add_argument('--C1_range', nargs=2, type=float, default=[-3, 4], metavar=('LB', 'UB'), help='log2 range of c1, used with --N (default: -3 4)')
    parser.add_argument('--C2_range', nargs=2, type=float, default=[-3, 4], metavar=('LB', 'UB'), help='log2 range of c2, used with --N (default: -3 4)')
    parser.add_argument('--B', type=int, default=1, help='Number of batch files to split the simulated datasets into (default: 1)')
    args = parser.parse_args()

    main(data_model=args.data_model, seed=args.seed,
         test_num=args.test_num, ID=args.ID, M1=args.M1, M2=args.M2,
         tree_file=args.tree_file, process=args.process,
         data_process=args.data_process, SW=args.SW,
         folder_dir=args.folder_dir, verbose=args.verbose,
         N=args.N, C1_range=args.C1_range, C2_range=args.C2_range,
         B=args.B)

"""
# defaults reproduce the final model's simulation, i.e. equivalent to:
python phyloKQL_simulate.py -f out_dir --data_model friedman --test_num 0.5 \
    --tree_file test_tree_v5.txt --ID test --M1 1 --M2 1 -p OU \
    --data_process BM --SW 5 --seed 12038 -v

# e.g. 10 x 10 (c1, c2) pairs with c1, c2 in [2^-3, 2^4], split in 56 batches:

python phyloKQL_simulate.py --data_model spiral\
        --N 56 --C1_range -3 4 --C2_range -3 4 --B 48 --SW 1\
        --folder_dir out_dir  --tree_file test_trees_all.txt
"""
