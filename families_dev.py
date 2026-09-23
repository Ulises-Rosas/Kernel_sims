import numpy as np
from multiclass_dev import PhyloKernel

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
                 copy_K_train = False,
                 fix_phi = False,
                 ) -> None:
        
        super().__init__(kernel = kernel, max_iter = max_iter,
                         lam = lam, warm_start = warm_start, 
                         tol = tol, verbose = verbose, 
                         copy_K_train= copy_K_train,
                         fix_phi=fix_phi,)

        self.reg = 'logistic'
        
        self.grad_tol = 1e-10
        # latent bounds
        self.ub =  10.0
        self.lb = -10.0
        self.hessian_tol = 1e-8

        self.Du_mu_func = self.get_Du_mu_internal
        self.Vu_func = self.get_Vu_func
        self.Du = np.array([])

    def mu(self, f):

        # f_tresh = 10
        # f_tresh = -np.log10(self.grad_tol)
        f = np.clip(f, self.lb, self.ub)
        mu =  1/(1 + np.exp(-f))
        return mu
    
    def get_Du_mu_internal(self, x):
        """
        Get mu vector (estimator) and D_u matrix with mu derivatives
        from the current function values using the logistic function
        """
        p1_out = self.mu(x)

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
        # u = np.clip(u, eps, 1 - eps)
        # u = self.mu(f_curr)
        return (1/self.phi) * (y * np.log(u) + (1 - y) * np.log(1 - u))
    
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

        f = np.clip(f, self.lb, self.ub)
        exp_f = np.exp(f - f.max(axis=1, keepdims=True))  # for numerical stability
        sum_exp_f = np.sum(exp_f, axis=1, keepdims=True)
        mu = exp_f / sum_exp_f
        return mu

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

    def Q(self, y, u):
        # u = self.mu(f_curr)
        return (1/self.phi) * np.sum(y * np.log(u), axis=1) # row-wise sum
