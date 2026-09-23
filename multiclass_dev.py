import numpy as np
from numpy import linalg as la

from phylokrr_dev.opt import R_u
from phylokrr_dev.agop import agop, RBF_kernel
from phylokrr.kernels import linear_kernel

from scipy.optimize import minimize

RED = "\033[31m"
RESET = "\033[0m"


def simple_centerK(K):
    """
    Center the kernel matrix K using the formula:
    K_centered = K - mean(K, axis=0) - mean(K, axis=1) + mean(K)

    It does no save means.
    Time complexity: O(n^2) for n x n matrix

    Previously this was used with 
    center_kernel(R_uc, save_means=False)

    """
    K_row_mean = np.mean(K, axis=1)
    K_col_mean = np.mean(K, axis=0)
    K_all_mean = np.mean(K)

    K_centered = K - K_col_mean - K_row_mean.reshape(-1, 1) + K_all_mean
    return K_centered



class PhyloKernel:
    """
    Phylogenetic kernel regression model with error correlation structure.
    """
    def __init__(self, 
                 kernel='rbf',
                 max_iter=300, 
                 lam=0.1,
                 warm_start = False,
                 tol = 0.001,
                 verbose = False,
                 copy_K_train = False,
                 fix_phi = False,
                 ) -> None:

        assert max_iter >= 1, "number of iterations should be a least 1" 
        self.reg = 'continous'
        self.process = "BM"
        self.inv_calculated = False
        self.use_quasi_newton = False
        self.memory_qn = 10
        self.do_corr = False # do correction or not
        self.corr_type = 'shrink'
        self.corr_scale_error = 'std'

        self.do_agop = False # do average gradient outer product or not
        self.agop_iter = 5
        self.M = None
        self.agop_diag = True

        # latent space bounds
        self.lb = -np.inf
        self.ub = np.inf
        
        self.Rnn = None

        self.alpha = np.array([])
        self.psc = 0.0 # phylo signal constant
        self.init_u = -2.0

        self.verbose = verbose
        self.warm_start = warm_start

        self.kernel = kernel
        self.lam = lam
        
        self.gamma = 0.1
        self.c = 0.1
        self.corr = 0.5 # correction for the conditional latent bias

        self.max_iter = max_iter
        self.tol = tol

        # speicify if the correlation matrix is needed
        # it might be the case that the input matrix is already
        # in form of a correlation matrix


        self.Du_mu_func = lambda x: (np.ones(x.shape[0]), x)
        self.Vu_func = lambda x: np.ones( x.shape[0] )
        
        self.X = np.array([])
        self.y = np.array([])

        
        self.copy_K = copy_K_train
        self.K_train = np.array([])
        self.K_col_mean = np.array([])
        self.K_all_mean = None


        self.phi = 1
        self.fix_phi = fix_phi
        self.r = np.array([]) # residuals
        self.dof = 1 # degrees of freedom
        self.L_inv = np.array([]) # R^-1/2


        self.f_curr = np.array([]) # current function values
        self.obj_curr = np.inf # objective function value

        self._verbose = False
        self.grad_tol = 1e-40
        self.hessian_tol = 1e-10


    def __repr__(self):
        return f"PhyloKernel(kernel={self.kernel}, max_iter={self.max_iter}, lam={self.lam}, warm_start={self.warm_start}, tol={self.tol})"
    
    def set_params(self, **params):
        """
        """
        for key, value in params.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                raise ValueError(f"Parameter {key} is not recognized.")

    def get_params(self):
        return  {'max_iter': self.max_iter, 
                'lam': self.lam, 
                'seed': self.seed,
                'intercept': self.alpha[0] if len(self.alpha) > 0 else None, 
                'tol': self.tol}

    def get_K(self, A, B, R = None, trainxtest=False, center = True):
        """
        Get the kernel matrix
        depending on the kernel type specified
        a parameter is passed to the kernel function
        Otherwise, the input matrix A is returned

        Parameters
        ----------
        A : np.ndarray
            input matrix
        B : np.ndarray
            input matrix

        Returns
        -------
        np.ndarray
            kernel
        """
        if self.kernel == 'rbf':

            K = RBF_kernel(A, B, self.gamma, M = self.M)

        elif self.kernel == 'linear':
            K =  linear_kernel(A, B, self.c)

        else:
            K =  A

        if R is not None:
            K += R

        if not center:
            return K

        else:
            K_row_mean = np.mean(K, axis=1)
            if trainxtest:

                assert len(self.K_col_mean) > 0, "K_col_mean is not set. Please set it before calling get_K with trainxtest=True"
                assert self.K_all_mean is not None, "K_all_mean is not set. Please set it before calling get_K with trainxtest=True"

                return K - self.K_col_mean - K_row_mean.reshape(-1, 1) + self.K_all_mean
            else:
                self.K_col_mean = np.mean(K, axis=0)
                self.K_all_mean = np.mean(K)
                return K - self.K_col_mean - K_row_mean.reshape(-1, 1) + self.K_all_mean
    
    
    def set_K_train(self, X):
        """
        Set the kernel matrix

        The general logic is as follows:

        if copy_S and S == 0:
            set_S

        elif not copy_S and S == 0:
            set_S (overwrite)

        elif copy_S and S != 0:
            do nothing

        elif not copy_S and S != 0:
            set_S (overwrite)        

        where S is Some matrix,
        in this case the kernel matrix

        Parameters
        ----------
        X : np.ndarray
            input matrix
        """

        if X is None:
            self.K_train = None
            return 

        if self.copy_K and len(self.K_train) != 0:
            pass

        else:
            Kc = self.get_K(X, X, R = self.Rnn, trainxtest=False, center = True)
            self.K_train = np.asfortranarray(Kc, dtype=float)


    def get_Du(self, f_curr):
        r"""
        Get mu vector (estimator) and W matrix with mu derivatives 
        with respect to f(x).


        Wii = \frac{\partial \mu}{\partial f(x)}
        if \mu is the identity function, then Wii = 1


        from the current function values.

        The function is will depend on the model being used.

        Parameters
        ----------
        f_curr : np.ndarray
            current function values

        Returns
        -------
        np.ndarray, np.ndarray
            Du, u
        """
        return self.Du_mu_func(f_curr)
    
    def get_Vu_sqrt_inv(self, f_curr):
        """
        get the square root of the inverse of 
        the variance of the estimator mu.

        the function is model dependent

        Parameters
        ----------
        f_curr : np.ndarray
            current function values

        Returns
        -------
        np.ndarray
            square root of the inverse of the variance
        """
        # Vu = self.Vu_func(f_curr)
        # Vu[Vu < self.epsilon] = self.epsilon
        # return Vu**(-1/2)
        return self.Vu_func(f_curr)**(-1/2)

    def get_V_Inv(self, f_curr):
        r"""
        Get the square root of the inverse of the V matrix
        V = A^1/2 R A^1/2 \phi
        A = variance of \mu
        R = correlation matrix
        phi = dispersion parameter

        V^(-1) = ( A^1/2  R      A^1/2  \phi)^-1
               =   A^-1/2 R^-1   A^-1/2 \phi^-1

        O(n^2) complexity

        Parameters
        ----------
        f_curr : np.ndarray
            current function values
        Returns
        -------
        np.ndarray
            V^(-1/2)
        """
        V_Inv = self.Vu_func(f_curr)**(-1)
        V_Inv *= (self.phi)**(-1) # O(n^2)
        return V_Inv

    def dR_R(self, u_curr, T):
        """
        Get the derivative of the correlation matrix with respect to u
        and the correlation matrix itself. Both are centered.

        """
        alpha_u = np.exp(-u_curr)
        R_uc = np.exp( -alpha_u * T)
        dR_uc = alpha_u * T * R_uc
        dR = simple_centerK(dR_uc,)
        R  = simple_centerK(R_uc, )
        return dR, R

    def R_dR_ddR_uc(self, u, T):
        """
        Get the correlation matrix, its first derivative and second derivative with respect to u
        and the correlation matrix itself. All are uncentered.
        """
        alpha_u = np.exp(-u)
        R_uc = np.exp( -alpha_u * T)
        dR_uc = alpha_u * T * R_uc
        ddR_uc = ( alpha_u * T  - np.ones(T.shape) ) * dR_uc
        return R_uc, dR_uc, ddR_uc

    # region: Quasi-newton optimization
    def opt_rij_QN(self, y, T, Kx, uc = 1):
        """
        Optimize the phylogenetic correlation parameter using gradient descent
        using Quasi-Newton method (L-BFGS-B)
        """
        n,m = self.get_dims(y)
        w_a = self.alpha.reshape(n + 1, m, order='F')

        w = w_a[1:, ]
        bias = w_a[0,:]

        def nabla_f(u_curr):
            dR, R = self.dR_R(u_curr, T)

            f_w = (Kx + R) @ w
            f_curr = f_w + bias

            Du, u = self.get_Du(f_curr)
            V_Inv = self.get_V_Inv(f_curr)

            g = 0.0
            for j in range(m):
                DuDa  = (Du[:,j][:,None] * dR) @ w[:,j]
                DuDaT_V = DuDa.T * V_Inv[:,j] # O(n^2)

                dp1 = (self.lam/2.) * (w[:,j].T @ dR @ w[:,j])
                g += -( 1./(n*m) )*( DuDaT_V @ (y[:,j] - u[:,j]) ) + dp1

            return g

        def f(u_curr):
            R_uc  = R_u(T, u_curr, center = False)
            R = simple_centerK(R_uc)
            f_w = (Kx + R) @ w # O(n^2)
            f_curr = f_w + bias

            suma = 0.0
            for j in range(m):
                suma += ( self.lam/2. ) * ( w[:,j].T @ f_w[:,j] )
            # p = (self.lam/2.) * (w.T @ f_w)

            obj = self.Obj(y, f_curr, suma)
            return obj

        result = minimize(
                    fun=f, 
                    x0=uc, 
                    method='L-BFGS-B',
                    tol = self.tol,
                    options={'maxiter': self.max_iter, 
                             'maxcor': self.memory_qn,
                             'maxls': 50,
                             'gtol': self.tol,
                             },
                    jac=nabla_f,
                    bounds=[(-4.0, 4.0)] 
                )

        if self.verbose:
            print(result)
            print(f"new phylo u: {result.x[0]}, new score: {result.fun}, old score: {f(uc)}")

        self.psc = result.x[0]
        conv = result.success
        if not conv:
            print(f"{RED}No convergence after %i iterations {RESET}" % self.max_iter)
            print(f"curr. alpha: {np.exp(-self.psc)}")

    def get_dims(self, y):
        nm = y.shape
        if len(nm) == 1:
            n = nm[0]
            m = 1
        else:
            n, m = nm            
        return n, m

    def opt_weights_QN(self, K, y):
        # K = self.K_train
        n, m = self.get_dims(y)

        p = K.shape[1] if K is not None else 0
        K_dtype = K.dtype if K is not None else float


        ones = np.ones(n, dtype=K_dtype)        
        if p > 0:
            K_0 = np.zeros((n + 1, p + 1), dtype=K_dtype)
            K_0[1:, 1:] = K
            lK_0 = self.lam * K_0

        else:
            lK_0 = np.array([[0]], dtype=K_dtype)


        # concatenate ones to K for intercept
        K_a = np.column_stack((ones, K)) if p > 0 else ones.reshape(-1, 1)
        K_a = np.ascontiguousarray(K_a, dtype=K_dtype)

        if not self.warm_start or np.any(np.isnan(self.alpha)):
            self.alpha = np.zeros( m * (p + 1) )
            self.alpha = np.ascontiguousarray(self.alpha, dtype=K_dtype)
            self.phi = 1

        assert len(self.alpha) > 0, "The model needs to be fitted first"

        def nabla_f(w_a: np.ndarray) -> np.ndarray:
            # w_a = self.alpha

            w_a = w_a.reshape(p + 1, m, order='F') # w matrix
            f_curr = K_a @ w_a
            # get derivatives of \mu and \mu
            Du, u = self.get_Du(f_curr)
            V_Inv = self.get_V_Inv(f_curr) # O(n^2)

            g = np.zeros((p + 1, m), dtype=K_dtype)

            for k in range(m):
                # k = 0
                DuDa = Du[:,k][:, None] * K_a # O(n^2)
                # print("V_Inv shape: ", V_Inv.shape)
                DuDaT_V = DuDa.T * V_Inv[:,k] # O(n^2)

                # gradient
                g[:,k] = (-1./(n*m))*( DuDaT_V @ (y[:,k] - u[:,k]) ) + lK_0 @ w_a[:,k] # O(n^2)
            
            return g.ravel(order='F')

            
        def f(w_a: np.ndarray) -> float:
            # w_a = self.alpha

            # f_curr = K_a @ w_a
            # pe = (self.lam/2.) * (w_a[1:].T @ K_a[:, 1:] @ w_a[1:])
            # obj =  self.Obj(y, f_curr, p = pe)

            w_a = w_a.reshape(p + 1, m, order='F')
            f_curr = K_a @ w_a
    
            suma = 0.0
            for k in range(m):
                suma += (self.lam/2.) * (w_a[1:, k].T @ K_a[:, 1:] @ w_a[1:, k])

            obj = self.Obj(y, f_curr, suma)
            return obj

        # O(mn^2) complexity mostly due to nabla_f and f evaluations
        # where m is the number of classes and n is the number of samples.
        # the optimization itself is O(qnm) where q is the history size of the L-BFGS-B algorithm
        # by default m = 10, so the overall complexity is O(mn^2)
        if self.verbose:
            print('Initial alpha: ', self.alpha[:4])
            print(f"Starting {self.solver} optimization with max_iter={self.max_iter}, tol={self.tol}, memory_qn={self.memory_qn}")
        if self.solver == 'L-BFGS-B':
            result = minimize(
                        fun=f, 
                        method='L-BFGS-B',
                        x0=self.alpha,
                        tol = self.tol,
                        options={'maxiter': self.max_iter, 
                                 'maxcor': self.memory_qn,
                                 'maxls': 50,   
                                 'gtol': self.tol,
                                },
                        jac=nabla_f,
                    )
        else:
            # O(n^2) complexity mostly due to nabla_f and f evaluations
            # the optimization itself is O((n*m)^2) 
            result = minimize(
                        fun=f, 
                        x0=self.alpha, 
                        method='BFGS',
                        tol = self.tol,
                        options={ 
                            'maxiter': self.max_iter,
                            'maxls': 50,   
                            'gtol': self.tol,
                            },
                        jac=nabla_f,
                    )

        conv = result.nit <= self.max_iter and result.success
        self.alpha = result.x
        self.f_curr = K_a @ self.alpha.reshape(p + 1, m, order='F')

        if self.verbose:
            print(result)
            print(self.mu(self.f_curr)[:4,:])
            print(y[:4,:])

        self.y = y.copy()

        if not conv:
            print(f"{RED}No convergence for weights after %i iterations {RESET}" % result.nit)
            print(f"{RED}gamma, phi, lam:, {self.gamma}, {self.phi}, {self.lam}{RESET}")
            print(f"{RED}first 4 weights: {self.alpha[:4]}{RESET}")
            
    # endregion

    def mu(self, f):
        return f

    def Q(self, y, u):
        return (1/self.phi) * -0.5 * (y - u)**2

    def Obj(self, y, f, p):
        # f = f_curr
        u = self.mu(f)
        return - np.mean(self.Q(y, u)) + p



    def get_new_score(self, K_x, y):
        """
        it uses self.K_train previously computed with the old u value

        Get the new score after updating the phylogenetic correlation parameter
        and update self.f_curr accordingly

        K_x: if it is None, then self.K_train is used for penalization term
        """

        n,m = self.get_dims(y)
        w_a = self.alpha.reshape(n + 1, m, order='F')

        f_wa = self.K_train @ w_a[1:,:]
        f_curr = f_wa + w_a[0,:]

        if K_x is not None:
            suma = 0.0
            for k in range(m):
                suma += (self.lam/2.) * (w_a[1:, k].T @ K_x @ w_a[1:, k])
        else:
            suma = 0.0
            for k in range(m):
                suma += (self.lam/2.) * (w_a[1:, k].T @ f_wa[:, k])

        return self.Obj(y, f_curr, p = suma)

    def update_K_train(self, u_new, T, K_x_uc):
        """
        Update uncentered Rnn and K_train
        """
        self.Rnn = R_u(T, u_new, center = False)
        K_uc = K_x_uc + self.Rnn

        K_row_mean = np.mean(K_uc, axis=1)
        self.K_col_mean = np.mean(K_uc, axis=0)
        self.K_all_mean = np.mean(K_uc)

        self.K_train = K_uc - self.K_col_mean - K_row_mean.reshape(-1, 1) + self.K_all_mean
        self.K_train = np.asfortranarray(self.K_train, dtype=float)

    # region: alternating optimization
    def step_fullK_opt(self, y, K_x, K_x_uc, vcv):

        self.opt_weights_QN(self.K_train, y)
        self.opt_rij_QN(y, T = vcv, Kx = K_x, uc = self.psc)
            
        self.update_K_train(u_new = self.psc, T = vcv, K_x_uc = K_x_uc)
    

    def fullK_opt(self, y, X, vcv):

        K_x_uc = self.get_K(X, X, R = None, trainxtest = False, center = False)
        K_x = simple_centerK(K_x_uc)

        self.update_K_train(u_new = self.psc, T = vcv, K_x_uc = K_x_uc)
        self.step_fullK_opt(y, K_x, K_x_uc, vcv)


        curr_score = self.get_new_score(None, y)
        curr_sig = self.psc
        curr_Kw = self.f_curr.copy()

        self.warm_start = True
        conv = False
        for i in range(self.max_iter):

            self.step_fullK_opt(y, K_x, K_x_uc, vcv)

            new_score = self.get_new_score(None, y)

            new_Kw = self.f_curr.copy()
            new_sig = self.psc

            lam_sq = 2*abs(curr_score - new_score)/max([abs(curr_score), abs(new_score), 1]) # a la scipy.minimize tol
            if self.verbose:
                print(f"iter: {i + 1}, phylo u: {self.psc}, lam_sq: {lam_sq}, new score: {new_score}, old score: {curr_score}, alpha: {np.exp(-self.psc)}")
            if  self.check_convergence(new_sig, curr_sig, 
                                      g_norm=self.grad_tol*2, 
                                      lam2=lam_sq, 
                                      i = i + 1) and  \
                self.check_convergence(new_Kw, curr_Kw, 
                                      g_norm=self.grad_tol*2, 
                                      lam2=lam_sq, 
                                      i = i  + 1):
                if self.verbose:
                    print(f"Converged at iteration {i + 1}, phylogenetic correlation parameter: {self.psc},\
                            new score: {new_score}, old score: {curr_score}, alpha: {np.exp(-self.psc)}")
                curr_score = new_score

                conv = True
                break

            curr_score = new_score
            curr_sig = new_sig
            curr_Kw = new_Kw

        if not conv:
            print(f"{RED}No convergence at alternating optimization. Phylogenetic correlation parameter: {self.psc}, new score: {new_score}, old score: {curr_score}, alpha: {np.exp(-self.psc)}{RESET}")

        self.obj_curr = curr_score
        self.warm_start = False

    # endregion: alternating optimization

    # region: fit
    def fit0(self, X, y, vcv = None):
        """
        Fit the model
        """

        self.psc = self.init_u # arbitrary initial value for the phylogenetic correlation parameter
        self.X = X # store the training data for prediction

        if self.process == 'BM':
            assert vcv is not None, "BM needs a correlation matrix as input"
            assert np.all(np.diag(vcv) != 0.0),\
                    "BM needs a correlation matrix"                
            self.Rnn = vcv
            self.set_K_train(self.X)
            self.opt_weights(self.K_train, y)

        elif self.process == 'OU':
            # initialize Rnn with u = self.init_u
            # uncentered
            assert vcv is not None, "OU needs path length matrix as input"
            assert np.all(np.diag(vcv) == 0.0),\
                    "OU needs path length matrix as input"

            n,m = self.get_dims(y)
            y = y.reshape(n, m, order='F')
            self.fullK_opt(y, X, vcv)

        elif self.process == 'none':
            self.set_K_train(self.X)
            self.opt_weights(self.K_train, y)

        else:
            raise ValueError("Model missing")

    def fit(self, X, y, vcv = None):

        self.fit0(X, y, vcv)

        if self.do_agop:
            if self.verbose:
                print("=========== Starting AGOP optimization")
            n, m = self.get_dims(y)
            W = self.alpha.reshape(n + 1, m, order='F')[1:,:]

            best_M = None
            best_score = np.inf

            for _ in range(self.agop_iter):

                Du, mu = self.get_Du(self.f_curr) # O(nm)
                Vu = self.get_V_Inv(self.f_curr) # O(nm)

                M = agop(Du, mu, Vu, self.y, W, self.lam,  
                         self.X, self.K_train, self.M, self.gamma,
                         diag = self.agop_diag)
                self.M = M / (M.max() + 1e-30)
                del M

                self.fit0(X, y, vcv)
                score = self.obj_curr

                if score < best_score:
                    best_score = score
                    best_M = self.M

                if self.verbose:
                    print(f"======= AGOP iter: {_ + 1}, score: {score}, best_score: {best_score}")
                    print("M: ")
                    print(self.M)

            # self.verbose = True
            self.M = best_M
            self.fit0(X, y, vcv)
    # endregion: fit


    def check_convergence(self, f_new, f_curr, g_norm = 1, lam2 = 1, i = 1):
        """
        Check convergence of the algorithm

        f_new: updated function values
        f_curr: current function values
        tol: tolerance level

        return: True if converged, False otherwise
        """
        if i == 0:
            return False

        # self.verbose = not True

        if lam2/2 <= self.tol:
            if self.verbose:
                print(f"Lambda squared is below the tolerance threshold (iter. {i})")
            return True

        if isinstance(f_new, np.ndarray) and isinstance(f_curr, np.ndarray):
            max_updt = np.linalg.norm(f_new - f_curr, ord = np.inf)
            w_max    = np.linalg.norm(f_new, ord = np.inf)
            if w_max == 0.0 or max_updt/w_max <= self.tol:
                if self.verbose:
                    print(f"f update is below the tolerance threshold (iter. {i})")
                return True

        else:
            max_updt = np.abs(f_new - f_curr)
            w_max = max([np.abs(f_new), np.abs(f_curr), 1])
            # w_max = np.abs(f_curr) + 1

            if max_updt/w_max <= self.tol:
            # if max_updt/max(w_max,1) <= self.tol:                
                if self.verbose:
                    print(f"f update is below the tolerance threshold (iter. {i})")
                return True


        if g_norm <= self.grad_tol:
            if self.verbose:
                print(f"Gradient norm is below the clipping threshold (iter. {i})")
            return True


        return False

    # region: prediction
    def correction_err_scale(self, e):

        if self.corr_scale_error == 'std':
            std = np.std(e)
            if std > 0:
                w = 1.0/std
            else:
                w = 1.0

        elif self.corr_scale_error == 'neff':
            n = len(e)
            off_diag_sum = (np.sum(self.Rnn) - np.trace(self.Rnn))
            if off_diag_sum > 0:
                neff = n**2 / (n + off_diag_sum)
            else:
                neff = n

            w = 1.0/neff

        elif self.corr_scale_error == 'n':
            w = 1.0/len(e)

        elif self.corr_scale_error == 'std_neff':
            n = len(e)
            off_diag_sum = (np.sum(self.Rnn) - np.trace(self.Rnn))
            if off_diag_sum > 0:
                neff = n**2 / (n + off_diag_sum)
            else:
                neff = n

            # neff = 2 if neff == 0 else 1
            if neff > 1:
                w = 1/np.linalg.norm(e, ord = 2)**2/(neff - 1)
            else:
                w = 1/np.linalg.norm(e, ord = 2)**2

        else:
            w = 1.0

        # print("w: ", w)
        return w

    def f_predict(self, X_test, Rzn = None):
        assert len(self.alpha) > 0, "The model needs to be fitted first"
        assert X_test is not None, "X_test should not be None"

        if self.K_train is None:
            f = self.alpha[0] * np.ones(X_test.shape[0])

        else:

            if Rzn is None or self.process == 'none':
                K_test = self.get_K(X_test, self.X, R = None, trainxtest = True)
                b, alpha = self.alpha[0], self.alpha[1:]
                return  K_test @ alpha + b

            if self.process == 'OU':
                Rzn = R_u(T = Rzn, u = self.psc, center = False)

            K_test = self.get_K(X_test, self.X, R = Rzn, trainxtest = True)
            b, alpha = self.alpha[0], self.alpha[1:]
            f =  K_test @ alpha + b # until here is ussually the way

            if not self.do_corr:
                return f
            else:
                tmp_f = np.clip(self.f_curr, self.lb, self.ub)
                eta_y = self.inv_af(self.y)
                e = eta_y - tmp_f
                e *= self.correction_err_scale(e)

                bias = Rzn @ la.solve( self.Rnn, e )

                if self.corr_type == 'convex':
                    return (1.0 - self.corr)*f + self.corr*bias
                
                elif self.corr_type == 'shrink':
                    f_zn = f + self.corr*bias
                    return f_zn

                else:
                    return f + bias


    def predict(self, X_test, R_12 = None):
        """
        Predict the response for identity link function

        Parameters
        ----------
        X_test : np.ndarray
            Test data

        Returns
        -------
        np.ndarray
            Predicted response
        """
        return self.f_predict(X_test, R_12)

    def inv_af(self, y):
        """
        inverse of the activation function
        """
        return y
    
    def conditional_latent(self, f_n, y, R_12, std = 1):
        e = (self.inv_af(y) - f_n)/std
        return R_12 @ np.linalg.solve(self.Rnn, e)
    # endregion: prediction

    def score(self, X_test, y_test, metric='rmse', R_12 = None):
        """
        The parameters already account for error correlation
        and the errors are assumed to be independent

        Parameters
        ----------
        X_test : np.ndarray
            test data

        y_test : np.ndarray
            test target

        metric : str, optional
            metric to use, by default 'rmse' and 'r2' for continuous
            targets and 'loss' for binary targets.

        R_12 : np.ndarray, optional
            Cross-correlation matrix between training and test data, by default None

        Returns
        -------
        float
            score
        """
        assert len(self.alpha) > 0, "The model needs to be fitted first"
        assert metric in ['rmse', 'loss', 'r2', 'log_loss'], "Invalid metric"
        
        # predict function is defined 
        # in the subclass
        y_pred = self.predict(X_test, R_12)

        if metric == 'rmse':
            return np.sqrt(np.mean((y_pred - y_test) ** 2))
                    
        else:
            u = ((y_test - y_pred) ** 2).sum()
            v = ((y_test - y_test.mean()) ** 2).sum()

            return 1 - (u / v)

