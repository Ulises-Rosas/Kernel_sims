import os
import numpy as np

# from phylokrr_dev.phyloK import Logistic
# from phylokrr_dev.agop import agop
# from phyloK_dev import Logistic, Multinomial

from families_dev import Multinomial
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from phylokrr_dev.block_cross_validation import k_fold_cv_weigthed




main_file_path = "/Users/ulises/Desktop/MLab/Kernels/Kernel_sims/Kernel_sims/out_dir"

X = np.load(f"{main_file_path}/simulated_multiclass_X.npy")
y = np.load(f"{main_file_path}/simulated_multiclass_y.npy")
n = X.shape[0]


np.random.seed(123)

train_idx = np.random.choice(X.shape[0], size=int(n*0.5), replace=False)
test_idx = np.array(list(set(range(n)) - set(train_idx)))
R = np.ones((n, n)) * 15
np.fill_diagonal(R, 0)




memory_qn = np.sqrt(X.shape[0]).astype(int)
# memory_qn = X.shape[0]
# memory_qn = X.shape[0]


# O(H*n + f + g),
# where H is the history and n is the number of variables.
# f is the time complexity of the function evaluation and 
# g is the time complexity of the gradient evaluation.


X_train, X_test = X[train_idx, :], X[test_idx, :]
y_train, y_test = y[train_idx], y[test_idx]
vcv_train = R[train_idx,:][:,train_idx]
vcv_test_12 = R[test_idx,:][:,train_idx]


self = Multinomial(kernel='rbf',)
self.max_iter = 100
self.memory_qn = np.sqrt(X.shape[0]*y.shape[1]).astype(int)
# self.memory_qn = X.shape[0]*y.shape[1]

print("lbfgs memory: ", self.memory_qn)
self.tol = 1e-6
self.process = 'OU'
self.seed = 12038
self.verbose = True
self.solver = 'L-BFGS-B'
# self.solver = 'BFGS'
self.gamma = 1
self.lam = 0.005
self.do_agop = True
self.agop_iter = 5
self.agop_diag = False



# y = y_train
# X = X_train
# vcv = vcv_train

# self.fit0(X, y, R)
# self.fit0(X_train, y_train, vcv_train)

self.fit(X_train, y_train, vcv_train)


# load multiclass regressor from sckit-learn
# import sklearn
# from sklearn.linear_model import LogisticRegression
