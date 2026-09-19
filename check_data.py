import os
import numpy as np

# from phylokrr_dev.phyloK import Logistic

from phylokrr_dev.agop import agop
from phyloK_dev import Logistic
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from phylokrr_dev.block_cross_validation import k_fold_cv_weigthed



# if __name__ == '__main__':

tree_file = './test_tree_v7.txt'
fix_phi = True
cv_sample = 100 # number of hyperparameters to test
cv_folds = 5
tol = 1e-6
cores = 5
max_iter = 100
test_num = 0.5
nested_loop_verbose = True
# do_adj = 'shrink' # adjust the level of correction for the latent space
do_adj = 'shrink' # the level of correction for the latent space
corr_scale_error = 'std'# if None, then the scale of the error is estimated from the data, otherwise it is set to this value
seed = 12038

np.random.seed(seed)

process = 'OU'
use_qn =  True
do_corr = True # do correction during cv
take_diag =  True

if not do_corr and do_adj:
    raise ValueError("Cannot adjust latent space without correction during CV")

if not do_adj:
    params = {
        'lam' : np.logspace(-9, 2, 100, base=2),
        'gamma'  : np.logspace(-5, 5, 100, base=2)
    }
    # params = {
    #     'lam' : np.logspace(-5, 15, 100, base=2),
    #     'gamma'  : np.logspace(-15, 3, 100, base=2),
    # }
    # params = {
    #     'lam' : np.array([1e-4, 1e-3, 1e-2, 1e-1, 1]),
    #     'gamma'  : np.array([1, 1/5, 1/10, 1/20]),
    # }    
else:
    params = {
        'lam' : np.logspace(-9, 2, 100, base=2),
        'gamma'  : np.logspace(-5, 5, 100, base=2),
        'corr' : np.logspace(-10, 0, 100, base=2)
    }
    # params = {
    #     'lam' : np.logspace(-5, 15, 100, base=2),
    #     'gamma'  : np.logspace(-15, 3, 100, base=2),
    #     'corr' : np.logspace(-10, 0, 100, base=2)
    #     # 'corr' : np.logspace(-5, 0, 100, base=2)
    # }
    # params = {
    #     'lam' : np.array([ 1e-3, 1e-2, 1e-1, 1]),
    #     'gamma'  : np.array([10, 5, 1, 1/5, 1/10, 1/20]),
    #     'corr' : np.logspace(-10, 0, 100, base=2)
    # }    

main_file_path = "./data_roja/picked_data"


ID = 'circle2_data'
i = 925.0 # c1:  1.3243463516588732 c2:  0.16712508671380788 
# i = 1310.0 # c1:  2.290242904264466 c2:  0.20400099160521443 split_weight:  5.0
# i = 3010.0
# all_params = np.load(main_file)


with open(tree_file, 'r') as f: 
    tree = f.read().strip()

# param = all_params[i]
# split_weight, c1, c2 = param
X = np.load(f"{main_file_path}/X_{ID}_{i}.npy")
y = np.load(f"{main_file_path}/y_{ID}_{i}.npy")
train_idx = np.load(f"{main_file_path}/train_idx_{ID}_{i}.npy")
test_idx = np.load(f"{main_file_path}/test_idx_{ID}_{i}.npy")
R = np.load(f"{main_file_path}/R_{ID}_{i}.npy")
W = np.load(f"{main_file_path}/W_{ID}_{i}.npy")

# n = X.shape[0]
# train_idx = np.random.choice(n, size = len(train_idx), replace=False)
# test_idx = np.array(list(set(range(n)) - set(train_idx)))


memory_qn = np.sqrt(X.shape[0]).astype(int)
# memory_qn = X.shape[0]


# O(H*n + f + g),
# where H is the history and n is the number of variables.
# f is the time complexity of the function evaluation and 
# g is the time complexity of the gradient evaluation.


X_train, X_test = X[train_idx, :], X[test_idx, :]
y_train, y_test = y[train_idx], y[test_idx]
vcv_train = R[train_idx,:][:,train_idx]
vcv_test_12 = R[test_idx,:][:,train_idx]



test_ratio =  np.sum(y_test == 1)/len(y_test)
train_ratio = np.sum(y_train == 1)/len(y_train)


if nested_loop_verbose:
    print()
    print('label_ratio train:', train_ratio)
    print('label_ratio test:', test_ratio)

    print('X_train shape: ', X_train.shape)
    print('X_test shape: ', X_test.shape)


if (not (0.2 < test_ratio < 0.8)) or (not (0.2 < test_ratio < 0.8)):
    raise ValueError("Test or train ratio is not between 0.2 and 0.8, please check the data split.")


X_train_offset = X_train.mean(axis=0)
X_train_scale = X_train.std(axis=0)

X_train -= X_train_offset
X_train /= X_train_scale

X_test -= X_train_offset
X_test /= X_train_scale

# X_train/= np.linalg.norm(X_train, axis=-1).reshape(-1, 1)
# X_test/= np.linalg.norm(X_test, axis=-1).reshape(-1, 1)


rng = np.random.default_rng(seed=seed)
lkrr = Logistic(get_corr= False, tol=tol, 
                max_iter=max_iter, kernel='rbf', 
                verbose= False, 
                fix_phi = fix_phi)  

lkrr.add_extraK = True
lkrr.process = process
lkrr.use_quasi_newton = use_qn
lkrr.memory_qn = memory_qn
lkrr.do_corr = do_corr
lkrr.corr_type = do_adj # ignore if do_corr is False, since it is not possible to adjust the latent space without correction during CV
lkrr.corr_scale_error = corr_scale_error
lkrr.init_u = -2.0


log_reg = LogisticRegression()
rf_model = RandomForestClassifier(n_estimators=100, random_state=seed)

#  lkrr.do_agop = True
# lkrr.agop_diag = take_diag
# import time
# start = time.time()
# lkrr_best_params = k_fold_cv_weigthed(
#     X_train, y_train, vcv_train,
#     lkrr,  W[train_idx], params,
#     folds=cv_folds,
#     sample=cv_sample, 
#     verbose= True, 
#     seed=seed,
#     cores = cores
# )


lkrr_best_params =   {'lam': np.float64(0.0005953125), 'gamma': np.float64(1.2936812840120273), 'corr': np.float64(0.0281345139217778)}


if nested_loop_verbose:
    # print('lkrr (+ phylo K) CV Time: ', time.time() - start
    print('lkrr best params: ', lkrr_best_params)

lkrr.set_params(**lkrr_best_params)
if nested_loop_verbose:
    print('ADDITIVE APPROACH:')

lkrr.do_agop = True
lkrr.agop_iter = 5
lkrr.fit(X_train, y_train, vcv_train)


if nested_loop_verbose:
    print('alpha param:', np.exp(-lkrr.psc))
    print('signal (meaningless if process is not OU):',lkrr.psc)


lkrr.do_corr = False
# just to check how it is doing without correction 
lkrr_probs = lkrr.predict_proba(X_test, vcv_test_12)
lkrr_loss = np.mean((lkrr_probs >= 0.5) == y_test)

if nested_loop_verbose:
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


# from sklearn.linear_model import LogisticRegression
# from sklearn.metrics.pairwise import pairwise_kernels


# # This measures similarity between every sample and every other sample
# K_train = pairwise_kernels(X_train, X_train, metric='rbf', gamma=1.0)

# klr_exact = LogisticRegression(C=1.0)
# klr_exact.fit(K_train, y_train)

# K_test = pairwise_kernels(X_test, X_train, metric='rbf', gamma=1.0)

# predictions = klr_exact.predict(K_test)
# # predictions.shape
# prediction_accuracy = np.mean(predictions == y_test)
# print("Standard Logistic Regression on RBF kernel accuracy:", prediction_accuracy)

plot_results = False

if plot_results:

    lkrr_predictions = (lkrr_probs >= 0.5).astype(int)
    lkrr_corr_predictions = (lkrr_probs_corr >= 0.5).astype(int)
    rf_predictions = rf_model.predict(X_test)

    plt.figure(figsize=(15, 5))
    plt.subplot(1, 3, 1)
    plt.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap='bwr', alpha=0.5, label='True labels')
    plt.title('True labels')
    plt.subplot(1, 3, 2)
    plt.scatter(X_test[:, 0], X_test[:, 1], c=lkrr_corr_predictions, cmap='bwr', alpha=0.5, label='LKRR corr predictions')
    plt.title('LKRR corr predictions')
    plt.subplot(1, 3, 3)
    plt.scatter(X_test[:, 0], X_test[:, 1], c=rf_predictions, cmap='bwr', alpha=0.5, label='RF predictions')
    plt.title('RF predictions')



    from sklearn.metrics import roc_curve, auc
    fpr, tpr, thresholds = roc_curve(y_test, lkrr_probs_corr)
    rf_fpr, rf_tpr, rf_thresholds = roc_curve(y_test, rf_model.predict_proba(X_test)[:, 1])
    roc_auc = auc(fpr, tpr)
    rf_roc_auc = auc(rf_fpr, rf_tpr)

    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot(rf_fpr, rf_tpr, color='blue', lw=2, label='RF ROC curve (area = %0.2f)' % rf_roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.01])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic')
    plt.legend(loc="lower right")

