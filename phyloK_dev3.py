"""
phyloK_dev3.py: phyloK_dev2.py with these fixes
  1. PhyloKernel (continuous) defines get_Du_mu_internal / get_Vu_func, so do_agop works.
  2. Correction error scaling: default None for continuous regression (conditional mean in
     y units); Logistic / Multinomial keep 'std'.
  3. AGOP uses the data kernel K_x only (the tree term r_lj does not depend on x),
     not K_train = K_x + R. Affects all classes.
  4. PhyloKernel.score has the same argument order as Logistic: score(X, y, R_12, metric);
     metric 'rmse' or 'r2' only.
  5. fullK_opt / AGOP efficiency and consistency:
     - the data kernel K_x built for K_train is kept (self.K_x) and reused by AGOP
       (no second RBF evaluation); it is dropped at the end of fit() so saved models stay small
     - f_curr is refreshed after the u step, so it matches the final (w, u)
       (it was computed with the previous u; used by AGOP and the correction)
     - opt_rij_QN: K_x @ w is computed once per u optimization, not per evaluation
     - opt_weights_QN: K^a = [1 K] and K_0 are no longer built ((n+1)^2 copies per call);
       2 matrix products per evaluation instead of 3
     - AGOP keeps the best fit in memory instead of refitting it at the end
  6. Minor: opt_weights no longer forces verbose on; conditional_latent uses _solve.
"""
import numpy as np
from numpy import linalg as la

from phylokrr_dev.opt import (newton_raphson_step_v2, backtracking_line_search_rij)
from phylokrr_dev.agop import agop
from phylokrr_dev.RBF import RBF_kernel
from phylokrr.kernels import linear_kernel

# from agop_dev import agop
from phylokrr_dev.tree_solve import solve_A_inv_b, get_tree_info, get_training_tree
from phylokrr.treeio import is_leaf

from scipy.special import expit, softmax
from scipy.optimize import minimize, minimize_scalar

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


def R_u(T, u, center = False):
    """
    Get the R matrix for the given T and u
    """
    # u = np.clip(u, -4, 4) # clip u to avoid numerical issues
    alpha = np.exp(-u)
    R_uc = np.exp( -alpha * T)
    return R_uc if not center else simple_centerK(R_uc)


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
                 get_corr = True,
                 copy_R_inv = False,
                 copy_K_train = False,
                 fix_phi = False,
                 ) -> None:

        assert max_iter >= 1, "number of iterations should be a least 1" 
        self.reg = 'continous'
        self.process = "BM"
        self.inv_calculated = False
        self.use_quasi_newton = True
        self.use_bracketing = False # for phylo signal optimization

        self.memory_qn = 10
        self.do_corr = False # do correction or not
        self.corr_type = 'shrink'
        self.corr_scale_error = None # continuous: no scaling (conditional mean in y units); 'std' for Logistic/Multinomial
        self.training_tree = None # for fast tree-based solve of R^-1 b, if available

        self.do_agop = False # do average gradient outer product or not
        self.agop_iter = 5
        self.M = None
        self.agop_diag = True
        self.agop_nogreedy = False

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
        # if False, the kernel matrices are used as-is (no centering)
        self.center_kernel = False
        
        self.gamma = 0.1
        self.c = 0.1
        self.corr = 0.5 # correction for the conditional latent bias

        self.max_iter = max_iter
        self.tol = tol

        # speicify if the correlation matrix is needed
        # it might be the case that the input matrix is already
        # in form of a correlation matrix
        self.get_corr = get_corr 

        self.Du_mu_func = self.get_Du_mu_internal
        self.Vu_func = self.get_Vu_func
        
        self.X = np.array([])
        self.y = np.array([])

        self.copy_R_inv = copy_R_inv
        self.R_inv = np.array([])
        
        self.copy_K = copy_K_train
        self.K_train = np.array([])
        self.K_x = None # data kernel (no tree term), kept during fit() for AGOP
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


    def _center(self, K):
        """Center K only if self.center_kernel is True; otherwise return K untouched."""
        return simple_centerK(K) if self.center_kernel else K

    def __repr__(self):
        return f"PhyloKernel(kernel={self.kernel}, max_iter={self.max_iter}, lam={self.lam}, warm_start={self.warm_start}, tol={self.tol}, center_kernel={self.center_kernel})"
    
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
                'intercept': self.alpha[0] if len(self.alpha) > 0 else None, 
                'tol': self.tol}

    def get_K(self, A, B, R = None):
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

        if not self.center_kernel:
            return K

        else:
            return simple_centerK(K)
    
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
            # data kernel kept for AGOP (r does not depend on x)
            self.K_x = self.get_K(X, X, R = None)
            Kc = self.K_x if self.Rnn is None else self.K_x + self.Rnn
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

        when R is the identity this is simply:
        V^(-1) = A^-1 \phi^-1

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
        and the correlation matrix itself. Centered only if self.center_kernel.

        """
        alpha_u = np.exp(-u_curr)
        R_uc = np.exp( -alpha_u * T)
        dR_uc = alpha_u * T * R_uc
        dR = self._center(dR_uc)
        R  = self._center(R_uc)
        return dR, R

    def R_dR_ddR_uc(self, u, T):
        """
        Get the correlation matrix, its first derivative and second derivative with respect to u
        and the correlation matrix itself. All are uncentered (centering is applied by the caller).
        """
        alpha_u = np.exp(-u)
        R_uc = np.exp( -alpha_u * T)
        dR_uc = alpha_u * T * R_uc
        ddR_uc = ( alpha_u * T  - np.ones(T.shape) ) * dR_uc
        return R_uc, dR_uc, ddR_uc

    def n_loss_terms(self, n, m):
        """
        Get N = number of loss terms, which is n*m for n samples and m classes
        """
        return n * m

    def get_DuV_Inv(self, f_curr):
        """
        Du * V^-1, the per-entry weight of (y - mu) in the gradients.
        Canonical links (Logistic, Multinomial) override this with 1/phi,
        which also avoids 0 * inf when mu saturates.
        """
            
        Du, _ = self.get_Du(f_curr)
        return Du * self.get_V_Inv(f_curr)    

    # region: Quasi-newton optimization
    def opt_rij_QN(self, y, T, Kx, uc = 1):
        """
        Optimize the phylogenetic correlation parameter using gradient descent
        using Quasi-Newton method (L-BFGS-B)
        """
        n, m = self.get_dims(y)
        W = self.alpha.reshape(n + 1, m, order='F')
        bias, w = W[0, :], W[1:, :]
        y_b = y.reshape(n, m, order='F')

        Kxw = Kx @ w # w is fixed during the u step: O(n^2 m) once, not per evaluation

        if self.use_bracketing:
            def my_f(u_curr):
                R = R_u(T, u_curr,  center = self.center_kernel)
                f_w = Kxw + R @ w
                f_curr = f_w + bias
                p = (self.lam/2.) * np.sum(w * f_w)
                return self.Obj(y_b, f_curr, p)

            result = minimize_scalar(my_f, bounds=(-4.0, 4.0), method='bounded', options={'xatol': 1e-6})
            self.psc = result.x

        else:
            N = self.n_loss_terms(n, m)
            def fnabla_f(u_curr):
                dR, R = self.dR_R(u_curr, T)
                f_w = Kxw + R @ w
                f_curr = f_w + bias

                u = self.mu(f_curr)
                DuV_inv = self.get_DuV_Inv(f_curr)

                dRw = dR @ w
                dp1 = (self.lam/2.) * np.sum(w * dRw)
                g = -np.sum(dRw * DuV_inv * (y_b - u) )/N + dp1

                p = (self.lam/2.) * np.sum(w * f_w)
                obj = self.Obj(y_b, f_curr, p)
                return obj, g
            
            result = minimize(
                        fun=fnabla_f, 
                        x0=uc, 
                        method='L-BFGS-B',
                        tol = self.tol,
                        options={ 'maxiter': self.max_iter,
                                'maxcor': 15, 'maxls': 50, 
                                'gtol': self.tol},
                        jac=True,
                        bounds=[(-4.0, 4.0)] 
                    )
            self.psc = result.x[0]


        conv = result.success
        if not conv:
            print(f"{RED}No convergence after %i iterations {RESET}" % self.max_iter)
            print(f"curr. alpha: {np.exp(-self.psc)}")
        self.obj_curr = result.fun

    def get_dims(self, y):
        nm = y.shape
        if len(nm) == 1:
            n = nm[0]
            m = 1
        else:
            n, m = nm            
        return n, m

    def opt_weights_QN(self, K, y):
        """
        Weights w~ = [b; w] for a fixed K with L-BFGS-B.
        K^a = [1 K] and K_0 are not built:
            K^a w~ = 1 b + K w,   lam K_0 w~ = [0; lam K w],   (K^a)^T r = [sum(r); K^T r]
        so each evaluation costs two n x n by n x m products (K w and K^T r).
        """
        n, m = self.get_dims(y)

        p = K.shape[1] if K is not None else 0
        K_dtype = K.dtype if K is not None else float

        if not self.warm_start or np.any(np.isnan(self.alpha)):
            self.alpha = np.zeros( m * (p + 1) ) # alpha weights
            self.alpha = np.ascontiguousarray(self.alpha, dtype=K_dtype)
            self.phi = 1

        assert len(self.alpha) > 0, "The model needs to be fitted first"

        y_b = y.reshape(n, m, order='F')
        N = self.n_loss_terms(n, m)

        def f_nablaf(w_a: np.ndarray) -> tuple[float, np.ndarray]:

            w_a = w_a.reshape(p + 1, m, order='F')
            b, w = w_a[0, :], w_a[1:, :]

            Kw = K @ w if p > 0 else np.zeros((n, m), dtype=K_dtype) # O(n^2 m)
            f_curr = Kw + b

            u = self.mu(f_curr)
            r = self.get_DuV_Inv(f_curr) * (y_b - u)

            g = np.empty((p + 1, m), dtype=K_dtype)
            g[0, :] = -r.sum(axis=0)/N                     # bias: not penalized
            if p > 0:
                g[1:, :] = -(K.T @ r)/N + self.lam * Kw   # O(n^2 m)
            suma = 0.5 * self.lam * np.sum(w * Kw)         # penalty term

            obj = self.Obj(y_b, f_curr, suma)
            return obj, g.ravel(order = 'F')
        # O(n^2) complexity mostly due to nabla_f and f evaluations
        # the optimization itself is O(nm) where m is the history size of the L-BFGS-B algorithm
        # by default m = 10, so the overall complexity is O(n^2)
        result = minimize(
                    fun=f_nablaf, 
                    x0=self.alpha, 
                    method='L-BFGS-B',
                    # method='BFGS',
                    tol = self.tol,
                    options={'maxiter': self.max_iter, 
                             'maxcor': self.memory_qn,
                             'maxls': 50,  
                             'gtol': self.tol,
                               },
                    jac=True,
                )

        # O(n^2) complexity mostly due to nabla_f and f evaluations
        # the optimization itself is O(n^2) 
        # result = minimize(
        #             fun=f, 
        #             x0=self.alpha, 
        #             # method='L-BFGS-B',
        #             method='BFGS',
        #             tol = self.tol,
        #             options={ 'maxiter': self.max_iter },
        #             jac=nabla_f,
        #         )
        
        conv = result.nit <= self.max_iter and result.success
        self.alpha = result.x
        W_a = self.alpha.reshape(p + 1, m, order='F')
        f_curr = (K @ W_a[1:, :] if p > 0 else 0.0) + W_a[0, :]
        self.f_curr = f_curr.ravel() if y.ndim == 1 else f_curr
        self.y = y.copy()

        if not conv:
            print(f"{RED}No convergence for weights after %i iterations {RESET}" % result.nit)
            print(f"{RED}gamma, phi, lam:, {self.gamma}, {self.phi}, {self.lam}{RESET}")
            print(f"{RED}first 4 weights: {self.alpha[:4]}{RESET}")

        self.obj_curr = result.fun
    # endregion

    # region: Newton-Raphson optimization
    def opt_weights(self, K, y, Kx = None):

        n = y.shape[0]
        p = K.shape[1] if K is not None else 0
        K_dtype = K.dtype if K is not None else float

        ones = np.ones(n, dtype=K_dtype)
        
        if p > 0:
            K_0 = np.zeros((n + 1, p + 1), dtype=K_dtype)
            K_x = K if Kx is None else Kx
            K_0[1:, 1:] = K_x
            lK_0 = self.lam * K_0

        else:
            lK_0 = np.array([[0]], dtype=K_dtype)


        # concatenate ones to K for intercept
        K_a = np.column_stack((ones, K)) if p > 0 else ones.reshape(-1, 1)
        K_a = np.ascontiguousarray(K_a, dtype=K_dtype)

        if not self.warm_start or np.any(np.isnan(self.alpha)):
            self.alpha = np.zeros(p + 1) # alpha weights
            self.alpha = np.ascontiguousarray(self.alpha, dtype=K_dtype)
            self.phi = 1

        assert len(self.alpha) > 0, "The model needs to be fitted first"

        def get_grad_Hessian(f_curr, w_a):
            # get derivatives of \mu and \mu
            Du, u = self.get_Du(f_curr)
            V_Inv = self.get_V_Inv(f_curr) # O(n^2)
    
            DuDa = Du[:,None] * K_a # O(n^2)

            # V_inv is diagonal
            # DuDaT_V = DuDa.T * np.diag(V_Inv) # O(n^2)
            DuDaT_V = DuDa.T * V_Inv # O(n^2)

            self.r = y - u # O(n)

            # gradient
            g =  (-1/n)*(DuDaT_V @ self.r) + lK_0 @ w_a # O(n^2)
            DuDaT_V_DuDa = np.matmul(DuDaT_V, DuDa) # O(n^3)
            H = (1/n)*(DuDaT_V_DuDa) + lK_0 # O(n^2)
            return H, g

        f_curr = K_a @ self.alpha
        conv = False
        for i in range(self.max_iter):

            H, G = get_grad_Hessian(f_curr, self.alpha)
            try:
                d = - np.linalg.solve( H, G ) # O(n^3)

            except np.linalg.LinAlgError:
                print("Linear algebra error occurred at solve. Adding hessian_tol to the diagonal of H and trying again.")
                d =  - np.linalg.solve( H + 1e-5 * np.eye(H.shape[0]), G ) # O(n^3)

            lam_sq = newton_raphson_step_v2(
                y, d, G, K_a, K_x, self.alpha,
                f = self.Obj, lam=self.lam
            )
    
            # new optimal function val
            f_new = K_a @ self.alpha
            # curr_acc = np.mean((self.mu(f_new) >= 0.5) == y)
            # print(f"Iter {i + 1}: curr_acc: {curr_acc}, lam_sq: {lam_sq}")
            if self.check_convergence(f_new, f_curr, 
                                      g_norm = np.linalg.norm(G, ord=2), 
                                      lam2 = lam_sq, 
                                      i = i+1):
                # print(f"Converge for alpha (iter. {i + 1})")
                conv = True
                f_curr = f_new
                break

            f_curr = f_new

            if self._verbose:
                print("Iteration: ", i)
                print('phi: ', self.phi)
                print('fix_phi: ', self.fix_phi)
                print()

        self.obj_curr = self.get_new_score(None, y)
        self.f_curr = f_curr
        self.y = y.copy()

        if not conv:
            print(f"{RED}No convergence after %i iterations {RESET}" % self.max_iter)
            print("gamma, phi, lam:")
            print(self.gamma, self.phi, self.lam)

    def opt_rij_gd_fullK(self, y, T, Kx, uc = 1):
        """
        Optimize the phylogenetic correlation parameter using gradient descent
        """
        n = y.shape[0]
        w = self.alpha[1:]
        bias = self.alpha[0]

        def grad_Hessian_wrt_u(u_curr):
            R_uc, dR_uc, ddR_uc = self.R_dR_ddR_uc(u_curr, T)

            ddR = self._center(ddR_uc)
            dR = self._center(dR_uc)
            R = self._center(R_uc)
                        
            f_w = (Kx + R) @ w
            f_curr = f_w + bias

            Du, u   = self.get_Du(f_curr)
            V_Inv   = self.get_V_Inv(f_curr)
            DuDa    = (Du[:,None] * dR) @ w
            DuDaT_V = DuDa.T * V_Inv # O(n^2)


            dp1 = (self.lam/2.) * (w.T @ dR @ w)
            dp2 = (self.lam/2.) * (w.T @ ddR @ w)
            g = -(1/n)*(DuDaT_V @ (y - u)) + dp1
            H  = (1/n)*(DuDaT_V @ DuDa) + dp2

            p = (self.lam/2.) * (w.T @ f_w)
            obj = self.Obj(y, f_curr, p)

            return g, H, obj

        u_curr = uc
        conv = False
        for i in range(self.max_iter):
            g, H, curr_obj = grad_Hessian_wrt_u(u_curr)

            d = -g/H if H > 0.0 else -g
            g_d = g * d
            lam_sq = -g_d

            step = backtracking_line_search_rij(
                y,  self.alpha, u_curr, T, Kx, d, g_d, 
                dist = self.reg, phi = self.phi, lam=self.lam,
                center = self.center_kernel
            )
            u_new = u_curr + (step*d)

            u_new = np.clip(u_new, -4.0, 4.0)  # Clip u_new to avoid extreme values
            if H <= 0.0:
                f_w_new = (Kx + R_u(T, u_new, center = self.center_kernel)) @ w
                new_obj = self.Obj(y, f_w_new + bias, (self.lam/2.) * (w.T @ f_w_new))
                lam_sq = 2*(curr_obj - new_obj)

            # print(f"Iter{i + 1}  ===== g: %f, H: %f, H^-1: %f, d: %f, u_curr: {u_curr}, u_new: {u_new}, alpha: {np.exp(-u_curr)}" %(g, H, H**(-1), d))
            if self.check_convergence(u_new, u_curr, 
                                      g_norm = np.abs(g), 
                                      lam2 = lam_sq,
                                      i = i + 1):
                if self.verbose:
                    print(f"Converge for r (iter. {i + 1})")
                    # print("---- CURR OBJ:", curr_obj)
                conv = True
                self.psc = u_new
                break

            u_curr = u_new

        if not conv:
            print(f"{RED}No convergence after %i iterations, curr. alpha %f {RESET}" % (self.max_iter, np.exp(-u_curr)))
            print(f"grad: %f, Hessian: %f, H^-1: %f, u_curr: {u_curr}, alpha: {np.exp(-u_curr)}" %(g, H, H**(-1)))
            # print(f"{RED}curr. alpha : {np.exp(-u_curr)}{RESET}")

        self.psc = u_curr
    # endregion

    def mu(self, f):
        return f

    def get_Du_mu_internal(self, f):
        """identity link: d mu / d f = 1, mu = f (used by AGOP)"""
        return np.ones_like(f), f

    def get_Vu_func(self, f_curr):
        """Gaussian variance function v(mu) = 1"""
        return np.ones_like(f_curr)

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
        # f_wa = self.K_train @ self.alpha[1:]
        # f_curr = f_wa + self.alpha[0]
        # if K_x is not None:
        #     p = (self.lam/2.) * (self.alpha[1:].T @ K_x @ self.alpha[1:])
        # else:
        #     p = (self.lam/2.) * (self.alpha[1:].T @ f_wa)
        # return self.Obj(y, f_curr, p = p)


        n,m = self.get_dims(y)
        w_a = self.alpha.reshape(n + 1, m, order='F')
        y_b = y.reshape(n, m, order='F')

        f_wa = self.K_train @ w_a[1:,:]
        f_curr = f_wa + w_a[0,:]

        if K_x is not None:
            suma = (self.lam/2.) * np.sum(w_a[1:,:] * (K_x @ w_a[1:,:])) # penalty term
            # suma = 0.0
            # for k in range(m):
            #     suma += (self.lam/2.) * (w_a[1:, k].T @ K_x @ w_a[1:, k])
            # p = (self.lam/2.) * (self.alpha[1:].T @ K_x @ self.alpha[1:])
        else:
            suma = (self.lam/2.) * np.sum(w_a[1:,:] * f_wa) # penalty term
            # suma = 0.0
            # for k in range(m):
            #     suma += (self.lam/2.) * (w_a[1:, k].T @ f_wa[:, k])
            # p = (self.lam/2.) * (self.alpha[1:].T @ f_wa)

        return self.Obj(y_b, f_curr, p = suma)

    def update_K_train(self, u_new, T, K_x_uc):
        """
        Update uncentered Rnn and K_train (K_train is centered only if self.center_kernel)
        """
        self.Rnn = R_u(T, u_new, center = False)
        # # print("Rnn at update K_train")
        # # print(self.Rnn[0:3, 0:3])
        K_uc = K_x_uc + self.Rnn
        if self.center_kernel:
            K_row_mean = np.mean(K_uc, axis=1)
            self.K_col_mean = np.mean(K_uc, axis=0)
            self.K_all_mean = np.mean(K_uc)
            self.K_train = K_uc - self.K_col_mean - K_row_mean.reshape(-1, 1) + self.K_all_mean
        else:
            self.K_train = K_uc
        self.K_train = np.asfortranarray(self.K_train, dtype=float)

    # region: alternating optimization
    def step_fullK_opt(self, y, K_x, K_x_uc, vcv):
        
        if self.use_quasi_newton or y.ndim > 1:

            self.opt_weights_QN(self.K_train, y)
            self.opt_rij_QN(y, T = vcv, Kx = K_x, uc = self.psc)
            
        else:
            self.opt_weights(self.K_train, y, Kx = None)
            self.opt_rij_gd_fullK(y, T = vcv, Kx = K_x, uc = self.psc)

        self.update_K_train(u_new = self.psc, T = vcv, K_x_uc = K_x_uc)
        # f_curr was computed by the weight step with the previous u
        self.refresh_f_curr(y)

    def refresh_f_curr(self, y):
        """f_curr = K_train w + b for the current weights and K_train, O(n^2 m)"""
        n, m = self.get_dims(y)
        W = self.alpha.reshape(n + 1, m, order='F')
        f = self.K_train @ W[1:, :] + W[0, :]
        self.f_curr = f.ravel() if y.ndim == 1 else f
    

    def fullK_opt(self, y, X, vcv):

        K_x_uc = self.get_K(X, X, R = None)
        K_x = self._center(K_x_uc)
        self.K_x = K_x_uc # data kernel kept for AGOP (r does not depend on x)

        self.update_K_train(u_new = self.psc, T = vcv, K_x_uc = K_x_uc)
        self.step_fullK_opt(y, K_x, K_x_uc, vcv)


        # curr_score = self.get_new_score(None, y)
        curr_score = self.obj_curr
        curr_sig = self.psc
        curr_Kw = self.f_curr.copy()

        self.warm_start = True
        conv = False
        for i in range(self.max_iter):

            self.step_fullK_opt(y, K_x, K_x_uc, vcv)

            # new_score = self.get_new_score(None, y)
            new_score = self.obj_curr

            new_Kw = self.f_curr.copy()
            new_sig = self.psc

            lam_sq = 2*abs(curr_score - new_score)/max([abs(curr_score), abs(new_score), 1]) # a la scipy.minimize tol
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

    def _opt_weights_fixedK(self, y):
        """
        Weights for a fixed kernel (BM / none). Newton for one output, as in
        phyloK_dev2; quasi-Newton for multiclass (Newton is single-output).
        """
        if y.ndim == 1:
            self.opt_weights(self.K_train, y)
        else:
            self.opt_weights_QN(self.K_train, y)
            # self.obj_curr = self.get_new_score(None, y)

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
            self._opt_weights_fixedK(y)

        elif self.process == 'OU':
            # initialize Rnn with u = self.init_u
            # uncentered
            assert vcv is not None, "OU needs path length matrix as input"
            assert np.all(np.diag(vcv) == 0.0),\
                    "OU needs path length matrix as input"

            self.fullK_opt(y, X, vcv)

        elif self.process == 'none':
            self.set_K_train(self.X)
            self._opt_weights_fixedK(y)

        else:
            raise ValueError("Model missing")

    def _state(self):
        """fitted state after fit0 (n x n matrices by reference: fit0 always builds new ones)"""
        return dict(alpha = np.array(self.alpha, copy=True), psc = self.psc,
                    f_curr = np.array(self.f_curr, copy=True), obj_curr = self.obj_curr,
                    y = self.y, M = self.M, K_train = self.K_train, Rnn = self.Rnn)

    def _set_state(self, state):
        for k, v in state.items():
            setattr(self, k, v)

    def fit(self, X, y, vcv = None):

        self.fit0(X, y, vcv)

        if self.do_agop:
            assert self.kernel == 'rbf', "AGOP is implemented for the RBF kernel"
            best_score = self.obj_curr
            best = self._state()   # fit without AGOP (M = None)
            n, m = self.get_dims(y)

            for _ in range(self.agop_iter):

                W = self.alpha.reshape(n + 1, m, order='F')[1:,:]
                if self.y.ndim == 1:
                    W = W[:, 0]

                Du, mu = self.get_Du_mu_internal(self.f_curr) # O(nm)
                Vu = self.get_Vu_func(self.f_curr) # O(nm)

                # the derivative of k = k_x + r w.r.t. x only involves k_x (r does not depend on x):
                # AGOP uses the data kernel of the current fit, kept by fit0 (no K_train = K_x + R)
                K_x = getattr(self, 'K_x', None)
                if K_x is None:
                    K_x = RBF_kernel(self.X, self.X, self.gamma, M = self.M)
                M = agop(Du, mu, Vu, self.y, W, self.lam,  
                         self.X, K_x, self.M, self.gamma,
                         diag = self.agop_diag)
                del K_x
                self.M = M / (M.max() + 1e-30)
                del M

                self.fit0(X, y, vcv)
                score = self.obj_curr

                if self.agop_nogreedy or score < best_score:
                    best_score = score
                    best = self._state()

            # fit0 is deterministic: restore the best fit instead of refitting it
            self._set_state(best)

        self.K_x = None # only needed by AGOP; keeps the saved model small
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

        self.verbose = not True

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

    # region: solve for bias correction
    def set_training_tree(self, nodes, leaves):
        """
        Set the training tree for bias correction
        
        Parameters
        ----------
        nodes : list or tuple
            List of nodes in the training tree. If tuple, it is assumed
            the it is the training tree tuple (nodes, root)

        leaves : list
            List of leaves names in the training tree
        """
        if nodes is None or leaves is None:
            self.training_tree = None
            return

        if isinstance(nodes, tuple) and len(nodes) == 2:
            # means that it already have the tree format
            # print("already pruned")
            self.training_tree = nodes
        else:
            self.training_tree = get_training_tree(nodes, leaves)

        self.training_names = leaves

    def _solve(self, E):

        if self.training_tree is None or self.process != 'OU':
            return la.solve(self.Rnn, E)

        new_nodes, new_root = self.training_tree
        children, parent, length, root, leaves = get_tree_info(new_root, new_nodes)
        name_of = {n.index: n.name for n in new_nodes if is_leaf(n)}
        row_of = {name: i for i, name in enumerate(self.training_names)}   # row order of X_train / e

        alpha = np.exp(-self.psc)
        E2 = E[:, None] if E.ndim == 1 else E

        out = np.empty(E2.shape)
        for k in range(E2.shape[1]):
            b = {v: E2[row_of[name_of[v]], k] for v in leaves}
            x = solve_A_inv_b(children, parent, length, alpha, root, b, leaves)
            
            for v in leaves:
                out[row_of[name_of[v]], k] = x[v]

        return out.ravel(order='F') if E.ndim == 1 else out
    # endregion: solve for bias correction

    # region: prediction
    def correction_err_scale(self, e):

        if self.corr_scale_error == 'std':
            std = np.std(e, axis=0)
            w = np.where(std > 0, 1.0 / np.where(std > 0, std, 1.0), 1.0)
            # w = 1/std

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

        # W is reshaped locally: self.alpha must stay flat for warm starts
        n, m = self.get_dims(self.y)
        W = self.alpha.reshape(n + 1, m, order='F')
        b, w = W[0, :], W[1:, :]
        # 1-D output for a single output (Logistic), (n_test, m) otherwise
        shape = lambda f: f.ravel() if self.y.ndim == 1 else f

        if self.K_train is None:
            return shape(np.tile(b, (X_test.shape[0], 1)))

        if Rzn is None or self.process == 'none':
            K_test = self.get_K(X_test, self.X, R = None)
            return shape(K_test @ w + b)

        if self.process == 'OU':
            Rzn = R_u(T = Rzn, u = self.psc, center = False)

        K_test = self.get_K(X_test, self.X, R = Rzn)
        f =  K_test @ w + b # until here is ussually the way

        if not self.do_corr:
            return shape(f)

        tmp_f = np.clip(self.f_curr, self.lb, self.ub)
        eta_y = self.inv_af(self.y)
        e = eta_y - tmp_f
        e = e * self.correction_err_scale(e)

        bias = (Rzn @ self._solve( e )).reshape(f.shape)

        if self.corr_type == 'convex':
            return shape((1.0 - self.corr)*f + self.corr*bias)
        
        elif self.corr_type == 'shrink':
            return shape(f + self.corr*bias)

        else:
            return shape(f + bias)

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
        return R_12 @ self._solve(e)
    # endregion: prediction

    def score(self, X_test, y_test, R_12 = None, metric = 'rmse'):
        """
        The parameters already account for error correlation
        and the errors are assumed to be independent

        Parameters
        ----------
        X_test : np.ndarray
            test data

        y_test : np.ndarray
            test target

        R_12 : np.ndarray, optional
            Cross-correlation matrix between training and test data, by default None

        metric : str, optional
            'rmse' (default; lower is better) or 'r2' (higher is better)

        Returns
        -------
        float
            score
        """
        assert len(self.alpha) > 0, "The model needs to be fitted first"
        assert metric in ['rmse', 'r2'], "Invalid metric for continuous regression: use 'rmse' or 'r2'"
        
        # predict function is defined 
        # in the subclass
        y_pred = self.predict(X_test, R_12)

        if metric == 'rmse':
            return np.sqrt(np.mean((y_pred - y_test) ** 2))
                    
        else:
            u = ((y_test - y_pred) ** 2).sum()
            v = ((y_test - y_test.mean()) ** 2).sum()

            return 1 - (u / v)

# region: Logistic regression
class Logistic(PhyloKernel):
    """
    Kernel Ridge Generalized Estimating Equations for 1 cluster.
    Using 1 cluster is what phylogenetic regresion deals with.

    This class is a subclass of KRGEE and implements the logistic function
    for the estimation of the mu vector and diagonal W matrix (derivative of mu).
    """
    def __init__(self,
                 kernel='rbf',
                 max_iter=300, 
                 lam=0.1,
                 warm_start = False,
                 tol = 0.001,
                 verbose = False,
                 get_corr = False,
                 copy_R_inv = False,
                 copy_K_train = False,
                 fix_phi = False,
                 ) -> None:
        
        super().__init__(kernel = kernel, max_iter = max_iter,
                          lam = lam, warm_start = warm_start, 
                          tol = tol, verbose = verbose, get_corr=get_corr, 
                          copy_R_inv=copy_R_inv,copy_K_train= copy_K_train, fix_phi=fix_phi,)

        self.reg = 'logistic'
        self.corr_scale_error = 'std'
        
        self.grad_tol = 1e-10
        # latent bounds
        self.ub =  10.0
        self.lb = -10.0
        self.hessian_tol = 1e-8

        self.Du_mu_func = self.get_Du_mu_internal
        self.Vu_func = self.get_Vu_func
        self.Du = np.array([])

    def mu(self, f):

        # f = np.clip(f, self.lb, self.ub)
        # return  1/(1 + np.exp(-f))
        return expit(f)
    
    def get_Du_mu_internal(self, x):
        """
        Get mu vector (estimator) and D_u matrix with mu derivatives
        from the current function values using the logistic function
        """
        p1_out = self.mu(x)

        # p1 * p0, derivative of mu estimator
        self.Du = p1_out * (1 - p1_out)
        # self.Du = p1_out * expit(-x)
        return self.Du, p1_out
    
    def get_Vu_func(self, f_curr):
        """
        Get the variance of the estimator mu
        """

        p1_out = self.mu(f_curr)
        self.Du = p1_out * (1 - p1_out)
        # self.Du = p1_out * expit(-f_curr)
        return self.Du
    
    def get_DuV_Inv(self, f_curr):
        r"""canonical (logit) link: 

        p1 * (1 - p1)  *  [ 1/( p1 * (1 - p1) * phi) ] = 1/phi
        """
        return np.full(np.shape(f_curr), 1.0/self.phi)

    def inv_af(self, y):
        """
        inverse of the activation function
        """
        # y_c = np.clip(y, self.hessian_tol, 1 - self.hessian_tol)
        # # y_c = np.clip(y, np.min(y), np.max([1 - 1e-3, np.max(y)]))
        # return np.log(y_c/(1 - y_c))
        y_c = np.zeros(len(y))
        y_c[y == 0] = self.lb
        y_c[y == 1] = self.ub
        return y_c

    
    def predict_proba(self, X_test, R_12=None):
        f = self.f_predict(X_test, R_12)
        return self.mu(f)
    
    def predict(self, X_test, R_12=None):
        """
        Predict the target values for the test data
        using the fitted model.

        Parameters
        ----------
        X_test : np.ndarray
            test data

        R_12 : np.ndarray, optional
            correlation matrix, by default None

        Returns
        -------
        np.ndarray
            predicted target values
        """
        mu = self.predict_proba(X_test, R_12)
        return (mu >= 0.5).astype(float)

    def Q(self, y, u, eps=1e-12):
        """
        Get the quasi-likelihood function for the logistic model
        """
        u = np.clip(u, eps, 1 - eps)
        # return (1/self.phi) *  (y * np.log(u) + (1 - y) * np.log1p(-u))
        return (1/self.phi) *  (y * np.log(u) + (1 - y) * np.log(1 - u))
    
    def score(self, X_test, y_test, R_12 = None, metric = 'log_loss', eps = 1e-8):
        """
        Score the model using the specified metric.
        Parameters
        ----------
        X_test : np.ndarray
            test data
        y_test : np.ndarray
            true labels
        R_12 : np.ndarray, optional
            correlation matrix, by default None
        metric : str, optional
            evaluation metric, by default 'log_loss'
        eps : float, optional
            small value to avoid division by zero, by default 1e-3

        Returns
        -------
        float
            evaluation score
        """

        if metric == 'loss':
            y_pred = self.predict(X_test, R_12)
            return np.mean(y_pred != y_test)
        
        elif metric == 'log_loss':
            y_pred_prob = self.predict_proba(X_test, R_12)
            # y_pred_prob = np.clip(y_pred_prob, eps, 1 - eps)
            # log_losses = -( y_test*np.log(y_pred_prob) + (1 - y_test)*np.log(1 - y_pred_prob) )

            log_losses = self.Q(y_test, y_pred_prob, eps=eps)
            return - np.mean( log_losses )
    
        else:
            raise ValueError("Invalid metric. Use 'loss' or 'log_loss'.")


# region: Multinomial regression
class Multinomial(PhyloKernel):
    """
    Kernel Ridge Generalized Estimating Equations for 1 cluster.
    Using 1 cluster is what phylogenetic regresion deals with.

    This class is a subclass of KRGEE and implements the multinomial function
    for the estimation of the mu vector and diagonal W matrix (derivative of mu).
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
        
        super().__init__(kernel = kernel, max_iter = max_iter,
                          lam = lam, warm_start = warm_start, 
                          tol = tol, verbose = verbose, 
                          copy_K_train= copy_K_train, fix_phi=fix_phi)

        self.reg = 'multinomial'
        self.corr_scale_error = 'std'
        
        self.grad_tol = 1e-10
        # latent bounds
        self.ub =  10.0
        self.lb = -10.0
        self.hessian_tol = 1e-8

        self.Du_mu_func = self.get_Du_mu_internal
        self.Vu_func = self.get_Vu_func
        self.Du = np.array([])

    def mu(self, f):
        """
        Calculate the multinomial probabilities for the given function values.
        """
        # f_tresh = 10
        # f_tresh = -np.log10(self.grad_tol)
        # f  = np.random.normal(1, 1, size = (3, 4))
        # f = self.K_train @ self.alpha.reshape(n + 1,m, order = "F")[1:,:]

        # f = np.clip(f, self.lb, self.ub)
        # exp_f = np.exp(f - f.max(axis=1, keepdims=True))  # for numerical stability
        # sum_exp_f = np.sum(exp_f, axis=1, keepdims=True)
        # mu = exp_f / sum_exp_f
        return softmax(f, axis=1)

    def get_Du_mu_internal(self, f):
        """
        Get mu vector (estimator) and D_u matrix with mu derivatives
        from the current function values using the multinomial function
        """
        p1_out = self.mu(f)
        # p1 * p0, derivative of mu estimator
        self.Du = p1_out * (1 - p1_out)
        return self.Du, p1_out

    def get_Vu_func(self, f_curr):
        """
        Get the variance of the estimator mu
        """
        p1_out = self.mu(f_curr)
        self.Du = p1_out * (1 - p1_out)
        return self.Du

    def get_DuV_Inv(self, f_curr):
        """
        canonical (softmax) link: the gradient of the log-likelihood w.r.t.
        f is (y - mu)/phi, so Du * V^-1 = 1/phi
        """
        return np.full(np.shape(f_curr), 1.0/self.phi)    

    def Q(self, y, u, eps=1e-12):
        u = np.clip(u, eps, 1.0 - eps)
        # m = y.shape[1]
        return (1/self.phi) * y * np.log(u)
        
    def inv_af(self, y):
        """
        inverse of the activation function
        """
        # y_c = np.clip(y, self.hessian_tol, 1 - self.hessian_tol)
        # # y_c = np.clip(y, np.min(y), np.max([1 - 1e-3, np.max(y)]))
        # return np.log(y_c/(1 - y_c))
        y_c = np.zeros(y.shape)
        y_c[y == 0] = self.lb
        y_c[y == 1] = self.ub
        return y_c

    def predict_proba(self, X_test, R_12=None):
        f = self.f_predict(X_test, R_12)
        return self.mu(f)
    
    def predict(self, X_test, R_12=None):
        """one-hot predicted class (same format as the training y)"""
        prob = self.predict_proba(X_test, R_12)
        out = np.zeros_like(prob)
        out[np.arange(prob.shape[0]), np.argmax(prob, axis=1)] = 1.0
        return out
        
    def score(self, X_test, y_test, R_12 = None, metric = 'log_loss', eps = 1e-8):
        """
        Score the model using the specified metric.
        Parameters
        ----------
        X_test : np.ndarray
            test data
        y_test : np.ndarray
            true labels
        R_12 : np.ndarray, optional
            correlation matrix, by default None
        metric : str, optional
            evaluation metric, by default 'log_loss'
        eps : float, optional
            small value to avoid division by zero, by default 1e-3

        Returns
        -------
        float
            evaluation score
        """

        if metric == 'loss':
            prob = self.predict_proba(X_test, R_12)
            return np.mean(np.argmax(prob, axis=1) != np.argmax(y_test, axis=1))
        
        elif metric == 'log_loss':
            y_pred_prob = self.predict_proba(X_test, R_12)
            # y_pred_prob = np.clip(y_pred_prob, eps, 1 - eps)
            # log_losses = -( y_test*np.log(y_pred_prob) + (1 - y_test)*np.log(1 - y_pred_prob) )

            log_losses = self.Q(y_test, y_pred_prob, eps=eps)
            return - np.mean( log_losses )
    
        else:
            raise ValueError("Invalid metric. Use 'loss' or 'log_loss'.")

#    def nabla_f(w_a: np.ndarray) -> np.ndarray:
#     # w_a = self.alpha

#     w_a = w_a.reshape(p + 1, m, order='F')
#     y_b = y.reshape(n, m, order='F')

#     f_curr = K_a @ w_a
#     # get derivatives of \mu and \mu
#     Du, u = self.get_Du(f_curr)
#     V_Inv = self.get_V_Inv(f_curr) # O(n^2)

#     g = np.zeros((p + 1, m), dtype=K_dtype)

#     for k in range(m):
#         # k = 0
#         DuDa = Du[:,k][:,None] * K_a # O(n^2)
#         # print("V_Inv shape: ", V_Inv.shape)
#         DuDaT_V = DuDa.T * V_Inv[:,k] # O(n^2)

#         # gradient
#         g[:,k] = (-1/(n*m))*(DuDaT_V @ (y_b[:,k] - u[:,k])) + lK_0 @ w_a[:,k] # O(n^2)

#     return g.ravel(order = 'F')

# def f(w_a: np.ndarray) -> float:
#     # w_a = self.alpha

#     # f_curr = K_a @ w_a
#     # pe = (self.lam/2.) * (w_a[1:].T @ K_a[:, 1:] @ w_a[1:])
#     # obj =  self.Obj(y, f_curr, p = pe)

#     w_a = w_a.reshape(p + 1, m, order='F') # w matrix
#     y_b = y.reshape(n, m, order='F') # big y, it does do anything if it is already in the right shape
#     f_curr = K_a @ w_a

#     suma = 0
#     for k in range(m):
#         suma += (self.lam/2.) * (w_a[1:, k].T @ K_a[:, 1:] @ w_a[1:, k])

#     obj = self.Obj(y_b, f_curr, suma)
#     return obj
