import numpy as np


def nabla_kw(l, K, X, M, gamma, w, diag = False):
    r"""
    Derivative of the kernel matrix with respect to the input data X.
    \partial k_{\ell, j} w_j / \partial x_{\ell} 

    total time complexity: 
    O(pnm + mp^2) = O( mp(n + p) ) if diag = False
    O(pnm)                         O\W
    """

    # w = W * Du[l,:]

    Dl = (X[l,:] - X).T #O(pn), R^{p,n}
    Dl_Kl = Dl * K[l,:] #O(pn), R^{p,n}
    Dl_kl_w = Dl_Kl @ w #O(pnm), R^{p,n} x R^{n,m} = R^{p,m}

    if M is None:
        return -2. * gamma * Dl_kl_w # O(pm), R^{p,m}
    
    else:
        if not diag:
            return -2. * gamma * M @ Dl_kl_w # O(mp^2), R^{p,m}
        else:
            assert len(M.shape) == 1, "M should be a vector when diag is True"
            b = -2. * gamma * M[:,None] * Dl_kl_w # O(pm), R^{p,m}
            # b = -2. * gamma * M * Dl_kl_w # O(pm), R^{p,m} # needed it for obj agop
            # print("b ", b)
            return b

def agop(Du, mu, Vu, y, W, lam, X, K, M, gamma, diag = False):
    """
    Outer product gradient for recursive feature machines

    Algorithm exact time complexity: 
    m be the number of outputs, 
    n be the number of samples, and 
    p be the number of features.

    If diag = False:
    O(nmp^2 + nmp(n + p)) = O(nmp(n + p)).  Assuming p < n, then O(mpn^2).
    
    
    If diag = True:
    O(n^2mp). regardless of the relationship between p and n.

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
    
    # mu = lkrr.mu(self.f_curr)
    # Vu = lkrr.Vu_func(self.f_curr)
    
    # lam = lkrr.lam
    # M = lkrr.M
    # gamma = lkrr.gamma
    
    # K = lkrr.K_train
    # X = X_train
    # W = lkrr.alpha[1:]
    # Du = lkrr.Du
    # y = lkrr.y
    # diag = True

    mn = y.shape
    d1_array = False
    if len(mn) == 1:
        n,m = (mn[0], 1)
        d1_array = True

    else:
        n,m = mn

    p = X.shape[1]

    if d1_array:
        W  = W[ :, np.newaxis]
        Du = Du[:, np.newaxis]


    G = np.zeros((p,m,n))
    # O( nmp(n + p) ) or O( n^2mp ) if diag
    for l in range(n):
        # l = 0
        # O( mp(n + p) ) or O( mnp ) if diag
        G[:,:,l] = nabla_kw(l, K, X, M, gamma,  W * Du[l,:], diag=diag)


    G -= G.mean(axis = 2, keepdims = True)

    if not diag:
        M_new = np.zeros((p,p), dtype=np.float32)
        # O(nmp^2)
        for l in range(n):
            # O(pm)
            nabla_f_l = G[:,:,l] # R^{p,m}
            # O(mp^2)
            # M_new += np.outer(nabla_f_l, nabla_f_l)
            M_new += nabla_f_l @ nabla_f_l.T
    else:
        M_new = np.zeros(p, dtype=np.float32)
        # O(nmp)
        for l in range(n):
            # O(mp)
            nabla_f_l = G[:,:,l]
            # O(mp)
            M_new += np.sum(nabla_f_l**2, axis = 1)

    return M_new / n


def obj_agop(Du, mu, Vu, y, W, lam, X, K, M, gamma, diag = False):
    """
    Average gradient outer product from the objective function

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
    # y =  np.random.normal(size = (10,1))

    mn = y.shape
    d1_array = False
    if len(mn) == 1:
        n,m = (mn[0], 1)
        d1_array = True

    else:
        n,m = mn

    # Du, mu = self.get_Du_mu_internal(f_curr) # O(nm)
    # Vu = self.get_Vu_func(f_curr) # O(nm)
    # yb = y.reshape(n, m, order='F') # O(nm)
    e = y - mu # O(nm)
    e /= Vu

    if d1_array:
        e  = e[ :, np.newaxis]
        W  = W[ :, np.newaxis]
        Du = Du[:, np.newaxis]

    c1 = m*n
    # O(nmp^2 + pmn^2) or O(mpn^2) if diag = True
    G = np.zeros((n, X.shape[1]))
    for l in range(n):
        nabla_f_l = np.zeros(X.shape[1])
        I_0 = np.ones(n)
        I_0[l] = 0
        # O(p^2m + pnm) or O(pnm) if diag = True
        for j in range(m):
            w1 = (-1./c1) * ( (W[:,j]*Du[l,j]*e[l,j]) + (W[l,j]*I_0*Du[:,j]*e[:,j]) )   # O(n)
            w2 = (lam/2.) * W[l,j] * ( W[:,j] + I_0 * W[:,j] ) # O(n)
            # O(p^2 + pn) or O(pn) if diag = True
            tmp = nabla_kw(l, K, X, M, gamma,  w1 + w2, diag=diag)
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