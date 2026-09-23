import numpy as np


def nabla_kw(l, K, X, M, gamma, wj, diag = False):
    r"""
    Derivative of the kernel matrix with respect to the input data X.
    \partial k_{\ell, j} w_j / \partial x_{\ell} 

    total time complexity: O(pp + pn) <= O(pn + pn) = O(pn) if p < n.
    """
    # X = X_train
    # K = lkrr.K_train
    # M = np.array([0.6442715, 1.       ])
    # gamma = lkrr.gamma
    # wj = lkrr.alpha[1:]
    # l = 0


    Dl = (X[l,:] - X).T #O(pn), R^{p,n}
    Dl_Kl = Dl * K[l,:] #O(pn), R^{p,n}
    Dl_kl_wj = Dl_Kl @ wj #O(pn), R^{p,1}

    if M is None:
        return -2 * gamma * Dl_kl_wj # O(p), R^{p,1}
    
    else:
        if not diag:
            return -2 * gamma * M @ Dl_kl_wj # O(p^2), R^{p,1}
        else:
            assert len(M.shape) == 1, "M should be a vector when diag is True"
            return -2 * gamma * M * Dl_kl_wj # O(p), R^{p,1}

def agop(Du, mu, Vu, y, W, lam, X, K, M, gamma, diag = False):
    """
    Outer product gradient for recursive feature machines

    Algorithm exact time complexity: 
    m be the number of outputs, 
    n be the number of samples, and 
    p be the number of features.

    If diag = False:
    O(nmp^2 + pmn^2 + np^2) = O(mnp^2 + mpn^2) = O(mnp(p + n)).  Assuming p < n, then O(mpn^2).
    
    
    If diag = True:
    O(nm + mpn^2 + np) = O(mpn^2) regardless of the relationship between p and n.

    Du: Derivative of the estimator, shape (n, m)
    mu: Estimator, shape (n, m)
    Vu: Variance of the estimator, shape (n, m)
    y: observed labels, shape (n, m)
    W: weights, shape (n, m)
    lam: regularization parameter
    X: input data, shape (n, p)
    K: kernel matrix, shape (n, n)
    M: prev agop matrix, shape (p, p) or shape (p,) if diag = True or None if no previous agop matrix is provided
    gamma: rbf kernel parameter
    diag: if True, then M is a vector and the agop matrix is computed as
          M_new = np.sum(nabla_f_l**2, axis=0) / n
          otherwise, M is a matrix and the agop matrix is computed as
          M_new = np.sum(np.outer(nabla_f_l, nabla_f_l), axis=0) / n
        

    returns: M_new, shape (p, p)

    reference:
        Supplemntary material of the paper "Mechanism for feature learning in neural networks and backpropagation-free machine learning models"
    """
    # f_curr = lkrr.f_curr
    # Du = lkrr.Du
    # mu = self.mu(self.f_curr)
    # Vu = lkrr.Vu
    # W = lkrr.alpha[1:]
    # lam = lkrr.lam
    # M = lkrr.M
    # gamma = lkrr.gamma
    # X = X_train
    # K = lkrr.K_train
    # self = lkrr
    # y = lkrr.y
    # diag = True

    mn = y.shape
    if len(mn) == 1:
        n,m = (mn[0], 1)

    else:
        n,m = mn

    # Du, mu = self.get_Du_mu_internal(f_curr) # O(nm)
    # Vu = self.get_Vu_func(f_curr) # O(nm)
    # yb = y.reshape(n, m, order='F') # O(nm)

    e = y - mu # O(nm)
    e /= Vu

    if m == 1:
        e  = e[:, np.newaxis]
        W  = W[:, np.newaxis]
        Du = Du[:, np.newaxis]

    c1 = m*n
    # O(nmp^2 + pmn^2) or O(mpn^2) if diag = True
    G = np.zeros((n, X.shape[1]))
    for l in range(n):
        nabla_f_l = np.zeros(X.shape[1])
        # O(p^2m + pnm) or O(pnm) if diag = True
        for j in range(m):
            d = (-1/c1) * Du[l,j] * e[l,j] + (lam/2) * W[l,j]
            # O(p^2 + pn) or O(pn) if diag = True
            tmp = nabla_kw(l, K, X, M, gamma, W[:,j], diag=diag) * d
            # print("tmp: ",tmp)
            nabla_f_l += tmp

        G[l,:] = nabla_f_l

    G -= G.mean(axis=0)
    if not diag:
        M_new = np.zeros((X.shape[1], X.shape[1]), dtype=np.float32)
        # O(np^2)
        for l in range(n):
            # O(p)
            nabla_f_l = G[l,:]
            # O(p^2)
            M_new += np.outer(nabla_f_l, nabla_f_l)
    else:
        M_new = np.zeros(X.shape[1], dtype=np.float32)
        # O(np)
        for l in range(n):
            # O(p)
            nabla_f_l = G[l,:]
            # O(p)
            M_new += nabla_f_l**2

    return M_new / n

def distance_matrix(a, b, M=None):
    """
    l2 norm squared matrix

    Time complexity:
    If M is full matrix, then O(np^2 + n^2p). Assuming p < n, then O(n^2p).
    If M is a diagonal or vector, then O(n^2p) regardless of the relationship between p and n.
    If M is None, then O(n^2p) regardless of the relationship between p and n.

    a: shape (m, p)
    b: shape (n, p)
    M: shape (p, p) or None
    
    returns: shape (m, n)
    """
    # a = X_test[:2, :]
    # b = X_train[:3, :]
    # # M = np.array([[2, 0], 
    # #               [0, 1]])
    # M = np.diag(M)

    # a = X_train
    # b = X_test
    # M = np.eye(a.shape[1])
    if M is None:
        # O(pn^2) for the whole matrix
        return np.linalg.norm(a[:, None, :] - b[None, :, :], axis=-1)**2
    
    else:
        # total time:
        # O(np + np + mnp + mn) = O(mnp) = O(n^2p) if m = n
        if len(M.shape) == 1:
            # O(np)
            A = ((a[ :, None, :] * M) @ a[:, :, None]).reshape(a.shape[0])
            # O(mp)
            B = ((b[ :, None, :] * M) @ b[:, :, None]).reshape(b.shape[0])
            # O(mpn + pn) = O(mnp)
            C = (a * M) @ b.T
            # O(mn)
            return np.add.outer(A,B) - 2*C


        else:
            # total time:
            # O(mn + mp^2 + np^2 + mpn + p^2n) 
            #  = O((m + n)p^2 + mpn)  
            # if m = n:
            # O(np^2 + n^2p) = O(npp + nnp) <= O(n^2p) if p < n

            # O(np^2)
            A = (a[ :, None, :] @ M @ a[:, :, None]).reshape(a.shape[0])
            # O(mp^2)
            B = (b[ :, None, :] @ M @ b[:, :, None]).reshape(b.shape[0])
            # O(mpn + p^2n)
            C = a @ M @ b.T
            # O(mn)
            return np.add.outer(A,B) - 2*C

def RBF_kernel(a, b, gamma, M = None):
    """
    Radial Basis Function

    Time complexity:
    If M is full matrix, then O(np^2 + n^2p). Assuming p < n, then O(n^2p).
    If M is a diagonal or vector, then O(n^2p) regardless of the relationship between p and n.
    If M is None, then O(n^2p) regardless of the relationship between p and n.

    """         
    a = np.ascontiguousarray(a, dtype=np.float64)
    b = np.ascontiguousarray(b, dtype=np.float64)
    # data sim
    D = distance_matrix(a, b, M=M)
    # D = np.where(D < 0, 0, D) # numerical stability
    tmp_rbf = -gamma * D

    np.exp(tmp_rbf, tmp_rbf) # RBF kernel. Inplace exponentiation
    return tmp_rbf
