"""
Implements all the logic for mixed logit models
"""
# pylint: disable=invalid-name
from .boxcox_functions import boxcox_param_deriv_mixed, \
                              boxcox_transformation_mixed
import scipy.stats
from scipy.optimize import minimize
from ._choice_model import ChoiceModel
from ._device import device as dev
import numpy as np
import itertools
import warnings

"""
Notations
---------
    N : Number of choice situations
    P : Number of observations per panel
    J : Number of alternatives
    K : Number of variables (Kf: fixed, Kr: random)
"""


class MixedLogit(ChoiceModel):
    """Class for estimation of Mixed Logit Models

    Attributes
    ----------
        coeff_ : numpy array, shape (n_variables + n_randvars, )
            Estimated coefficients

        coeff_names : numpy array, shape (n_variables + n_randvars, )
            Names of the estimated coefficients

        stderr : numpy array, shape (n_variables + n_randvars, )
            Standard errors of the estimated coefficients

        zvalues : numpy array, shape (n_variables + n_randvars, )
            Z-values for t-distribution of the estimated coefficients

        pvalues : numpy array, shape (n_variables + n_randvars, )
            P-values of the estimated coefficients

        loglikelihood : float
            Log-likelihood at the end of the estimation

        convergence : bool
            Whether convergence was reached during estimation

        total_iter : int
            Total number of iterations executed during estimation

        estim_time_sec : float
            Estimation time in seconds

        sample_size : int
            Number of samples used for estimation

        aic : float
            Akaike information criteria of the estimated model

        bic : float
            Bayesian information criteria of the estimated model
    """

    def __init__(self):
        """Init Function"""
        super(MixedLogit, self).__init__()
        self.rvidx = None  # Boolean index of random vars in X. True = rand var
        self.rvdist = None

    # X: (N, J, K)
    def fit(self, X, y, varnames=None, alts=None, isvars=None, transvars=None,
            ids=None, transformation=None, weights=None, avail=None,
            randvars=None, panels=None, base_alt=None, fit_intercept=False,
            init_coeff=None, maxiter=2000, random_state=None, correlation=None,
            n_draws=200, halton=True, verbose=1, method="bfgs"):
        """Fit Mixed Logit models.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_variables)
            Input data for explanatory variables in long format

        y : array-like, shape (n_samples,)
            Choices (outcome) in long format

        varnames : list, shape (n_variables,)
            Names of explanatory variables that must match the number and
            order of columns in ``X``

        alts : array-like, shape (n_samples,)
            Alternative indexes in long format or list of alternative names

        isvars : list
            Names of individual-specific variables in ``varnames``

        transvars: list, default=None
            Names of variables to apply transformation on

        ids : array-like, shape (n_samples,)
            Identifiers for choice situations in long format.

        transformation: string, default=None
            Name of transformation to apply on transvars

        weights : array-like, shape (n_variables,), default=None
            Weights for the choice situations in long format.

        avail: array-like, shape (n_samples,)
            Availability of alternatives for the choice situations. One when
            available or zero otherwise.

        randvars : dict
            Names (keys) and mixing distributions (values) of variables that
            have random parameters as coefficients. Possible mixing
            distributions are: ``'n'``: normal, ``'ln'``: lognormal,
            ``'u'``: uniform, ``'t'``: triangular, ``'tn'``: truncated normal

        panels : array-like, shape (n_samples,), default=None
            Identifiers in long format to create panels in combination with
            ``ids``

        base_alt : int, float or str, default=None
            Base alternative

        fit_intercept : bool, default=False
            Whether to include an intercept in the model.

        init_coeff : numpy array, shape (n_variables,), default=None
            Initial coefficients for estimation.

        maxiter : int, default=200
            Maximum number of iterations

        random_state : int, default=None
            Random seed for numpy random generator

        correlation: boolean or list, default=None
            If boolean finds correlation for all random (non trans) vars
            If list finds correlation between variables specified

        n_draws : int, default=200
            Number of random draws to approximate the mixing distributions of
            the random coefficients

        halton : bool, default=True
            Whether the estimation uses halton draws.

        verbose : int, default=1
            Verbosity of messages to show during estimation. 0: No messages,
            1: Some messages, 2: All messages

        method: string, default="bfgs"
            specify optimisation method


        Returns
        -------
        None.
        """

        X, y, initialData, varnames, alts, isvars, transvars, ids, weights, panels, avail\
            = self._as_array(X, y, varnames, alts, isvars, transvars, ids,
                             weights, panels, avail)

        self._validate_inputs(X, y, alts, varnames, isvars, ids, weights, panels,
                              base_alt, fit_intercept, maxiter)
        self._pre_fit(alts, varnames, isvars, transvars, base_alt,
                      fit_intercept, transformation, maxiter, panels,
                      correlation, randvars)
        self.randvarsdict = randvars
        #  random variables not transformed
        self.randvars = [x for x in self.randvars if x not in transvars]
        #  random variables that are transformed
        self.randtransvars = [x for x in transvars if (x in randvars) and
                                                      (x not in self.randvars)]
        self.fixedtransvars = [x for x in transvars if x not in
                               self.randtransvars]
        self.n_draws = n_draws
        # divide the variables in varnames into fixed, fixed transformed,
        # random, random transformed by getting 4 index arrays
        # also for random and random transformed save the distributions
        # in a separate array
        self.fxidx, self.fxtransidx = [], []
        self.rvidx, self.rvdist = [], []
        self.rvtransidx, self.rvtransdist = [], []
        for var in self.varnames:
            with warnings.catch_warnings():
                # CURRENTLY IGNORING FUTURE WARNING
                # CURRENT PY: 3.8.3, numpy: 1.18.5
                warnings.simplefilter(action='ignore', category=FutureWarning)
                if isvars is not None:
                    if var in isvars:
                        continue
            if var in randvars.keys():
                self.fxidx.append(False)
                self.fxtransidx.append(False)
                if var in self.randvars:
                    self.rvidx.append(True)
                    self.rvdist.append(randvars[var])
                    self.rvtransidx.append(False)
                else:
                    self.rvidx.append(False)
                    self.rvtransidx.append(True)
                    self.rvtransdist.append(randvars[var])
            else:
                self.rvidx.append(False)
                self.rvtransidx.append(False)
                self.rvdist.append(False)
                self.rvtransdist.append(False)
                if var in transvars:
                    self.fxtransidx.append(True)
                    self.fxidx.append(False)
                else:
                    self.fxtransidx.append(False)
                    self.fxidx.append(True)

        self.rvidx = np.array(self.rvidx)
        self.rvtransidx = np.array(self.rvtransidx)
        self.fxidx = np.array(self.fxidx)
        self.fxtransidx = np.array(self.fxtransidx)
        if random_state is not None:
            np.random.seed(random_state)
        self.initialData = initialData
        X, y, panels = self._arrange_long_format(X, y, ids, alts, panels)
        y = y
        X, Xnames = self._setup_design_matrix(X)
        J, K, R = X.shape[1], X.shape[2], n_draws

        if self.transvars is not None and self.transformation is None:
            # if transvars provided and no specified transformation function
            # give default to boxcox
            self.transformation = "boxcox"

        if self.transformation == "boxcox":
            self.transFunc = boxcox_transformation_mixed
            self.transform_deriv = boxcox_param_deriv_mixed

        R = n_draws
        if panels is not None:  # If panels
            X, y, panel_info = self._balance_panels(X, y, panels)
            N, P = panel_info.shape
        else:
            N, P = X.shape[0], 1
            panel_info = np.ones((N, 1))

        X = X.reshape(N, P, J, K)
        y = y.reshape(N, P, J, 1)

        self.n_draws = n_draws
        self.n_draws = n_draws
        self.verbose = verbose
        self.total_fun_eval = 0

        if weights is not None:
            weights = weights*(N/np.sum(weights))  # Normalize weights

        if avail is not None:
            avail = avail.reshape(N, J)

        # Generate draws
        draws, drawstrans = self._generate_draws(self.N, R, halton)  # (N,Kr,R)
        # 2x Kftrans - mean and lambda, 3x Krtrans - mean, s.d., lambda
        # Kchol, Kbw - relate to random variables, non-transformed
        # Kchol - cholesky matrix, Kbw the s.d. for random vars
        n_coeff = self.Kf + self.Kr + self.Kchol + self.Kbw + 2*self.Kftrans +\
            3*self.Krtrans
        if init_coeff is None:
            betas = np.repeat(.1, n_coeff)
        else:
            betas = init_coeff
            if len(init_coeff) != n_coeff:
                raise ValueError("The size of init_coeff must be: " + n_coeff)
        if dev.using_gpu:
            X, y = dev.to_gpu(X), dev.to_gpu(y)
            panel_info = dev.to_gpu(panel_info)
            draws = dev.to_gpu(draws)
            drawstrans = dev.to_gpu(drawstrans)
            if weights is not None:
                weights = dev.to_gpu(weights)
            if avail is not None:
                avail = dev.to_gpu(avail)
            if verbose > 0:
                print("Estimation with GPU processing enabled.")
        positive_bound = (0, 1e+30)
        any_bound = (-1e+30, 1e+30)
        corr_bound = (-1, 1)
        lmda_bound = (-5, 5)
        
        # to be used with L-BFGS-B method, automatically generate bounds
        bound_dict = {
            "bf": (any_bound, self.Kf),
            "br_b": (any_bound, self.Kr),
            "chol": (corr_bound, self.Kchol),
            "br_w": (positive_bound, self.Kr - self.correlationLength),
            "bf_trans": (any_bound, self.Kftrans),
            "flmbda": (lmda_bound, self.Kftrans),
            "br_trans_b": (any_bound, self.Krtrans),
            "br_trans_w": (positive_bound, self.Krtrans),
            "rlmbda": (lmda_bound, self.Krtrans)
        }

        # list comrephension to add number of bounds for each variable type
        # bound[1][0] - the bound range
        # bound[1][1] - how many bounds to add
        bnds = [(bound[1][0],) * int(bound[1][1])
                for bound in bound_dict.items() if bound[1][1] > 0]
        # convert into appropriate format for L-BFGS-B
        bnds = [tuple(itertools.chain.from_iterable(bnds))][0]

        if method == 'L-BFGS-B':
            optimizat_res = \
                minimize(
                    self._loglik_gradient,
                    betas,
                    jac=True,
                    method=method,
                    args=(X, y, panel_info, draws, drawstrans, weights,
                          avail),
                    tol=1e-5,
                    bounds=bnds,
                    options={
                        'gtol': 1e-4,
                        'maxiter': maxiter,
                        'disp': verbose > 0
                    }
                )
        else:
            optimizat_res = \
                minimize(
                    self._loglik_gradient,
                    betas,
                    jac=True,
                    method=method,
                    args=(X, y, panel_info, draws, drawstrans,
                          weights, avail),
                    tol=1e-5,
                    options={
                        'gtol': 1e-4,
                        'maxiter': maxiter,
                        'disp': verbose > 0
                    }
                )
        self._post_fit(optimizat_res, Xnames, N, verbose)

    def _compute_probabilities(self, betas, X, panel_info, draws, drawstrans,
                               avail):
        """Compute the standard logit-based probabilities.

        Random and fixed coefficients are handled separately.
        """
        Bf, Br = self._transform_betas(betas, draws)  # Get fixed and rand coef
        X = X.reshape((self.N, self.P, self.J, self.R))
        Xf = X[:, :, :, ~self.rvidx]  # Data for fixed coefficients
        Xr = X[:, :, :, self.rvidx]   # Data for random coefficients
        V = np.zeros_like(X, dtype=float)
        if (len(Bf) > 0):
            XBf = dev.np.einsum('npjk,k -> npj', Xf, Bf)  # (N,P,J)
            V = XBf[:, :, :, None]
        if (len(Br) > 1):
            XBr = dev.np.einsum('npjk,nkr -> npjr', Xr, Br)  # (N,P,J,R)
            V += XBr  # (N,P,J,R)
        V[V > 700] = 700
        eV = dev.np.exp(V)
        if avail is not None:
            eV = eV*avail[:, None, :, None]  # Acommodate availablity of alts.
        sumeV = dev.np.sum(eV, axis=2, keepdims=True)
        sumeV[sumeV == 0] = 1e-30
        p = eV/sumeV  # (N,P,J,R)
        p = p*panel_info[:, :, None, None]  # Zero for unbalanced panels
        return p

    def _loglik_gradient(self, betas, X, y, panel_info, draws, drawstrans,
                         weights, avail):
        """Compute the log-likelihood and gradient.

        Fixed and random parameters are handled separately to
        speed up the estimation and the results are concatenated.
        """
        # Segregating initial values to fixed betas (Bf), 
        # random beta means (Br_b)
        # for both non-transformed and transformed variables
        # and random beta cholesky factors (chol)
        if dev.using_gpu:
            betas = dev.to_gpu(betas)

        beta_segment_names = ["Bf", "Br_b", "chol", "Br_w", "Bftrans",
                              "flmbda", "Brtrans_b", "Brtrans_w", "rlmda"]
        var_list = dict()
        iterations = [self.Kf, self.Kr, self.Kchol, self.Kbw, self.Kftrans,
                      self.Kftrans, self.Krtrans, self.Krtrans, self.Krtrans]
        i = 0
        for count, iteration in enumerate(iterations):
            prev_index = i
            i = int(i + iteration)
            var_list[beta_segment_names[count]] = betas[prev_index:i]

        Bf, Br_b, chol, Br_w, Bftrans, flmbda, Brtrans_b, Brtrans_w, rlmda = \
            var_list.values()
        if dev.using_gpu:
            Bf, Br_b, chol, Br_w, Bftrans, flmbda, Brtrans_b, \
                Brtrans_w, rlmda = \
                dev.to_gpu(Bf), dev.to_gpu(Br_b), dev.to_gpu(chol), \
                dev.to_gpu(Br_w), dev.to_gpu(Bftrans), dev.to_gpu(flmbda), \
                dev.to_gpu(Brtrans_b), dev.to_gpu(Brtrans_w), dev.to_gpu(rlmda)
        # creating cholesky matrix for the variance-covariance matrix
        # all random variables not included in correlation will only
        # have their standard deviation computed
        chol_mat = np.zeros((self.correlationLength, self.correlationLength))
        indices = np.tril_indices(self.correlationLength)
        chol_mat[indices] = chol
        chol_mat_temp = np.zeros((self.Kr, self.Kr))
        chol_mat_temp[:self.correlationLength, :self.correlationLength] = \
            chol_mat

        for i in range(self.Kr - self.correlationLength):
            chol_mat_temp[i+self.correlationLength,
                          i+self.correlationLength] = \
                Br_w[i]
        chol_mat = chol_mat_temp
        # Creating random coeffs using Br_b, cholesky matrix and random draws
        # Estimating the linear utility specification (U = sum of Xb)
        V = np.zeros((self.N, self.P, self.J, self.n_draws))
        Xf = X[:, :, :, self.fxidx]
        Xftrans = X[:, :, :, self.fxtransidx]
        Xr = X[:, :, :, self.rvidx]
        Xrtrans = X[:, :, :, self.rvtransidx]

        if self.Kf != 0:
            XBf = np.einsum('npjk,k -> npj', Xf, Bf)
            V += XBf[:, :, :, None]*self.S[:, :, :, None]
        if self.Kr != 0:
            Br = Br_b[None, :, None] + np.matmul(chol_mat, draws)
            Br = self._apply_distribution(Br, self.rvdist)
            XBr = np.einsum('npjk, nkr -> npjr', Xr, Br)  # (N, P, J, R)
            V += XBr*self.S[:, :, :, None]
        #  transformation
        #  transformations for variables with fixed coeffs
        if self.Kftrans != 0:
            Xftrans_lmda = self.transFunc(Xftrans, flmbda)
            Xftrans_lmda[np.isneginf(Xftrans_lmda)] = -1e+30
            Xftrans_lmda[np.isposinf(Xftrans_lmda)] = 1e+10
            # Estimating the linear utility specificiation (U = sum XB)
            Xbf_trans = np.einsum('npjk,k -> npj', Xftrans_lmda, Bftrans)
            # combining utilities
            V += Xbf_trans[:, :, :, None]

        # transformations for variables with random coeffs
        if self.Krtrans != 0:
            # creating the random coeffs
            Brtrans = Brtrans_b[None, :, None] + \
                    drawstrans[:, 0:self.Krtrans, :] * Brtrans_w[None, :, None]
            Brtrans = self._apply_distribution(Brtrans, self.rvtransdist)
            # applying transformation
            Xrtrans_lmda = self.transFunc(Xrtrans, rlmda)
            Xrtrans_lmda[np.isposinf(Xrtrans_lmda)] = 1e+10
            Xrtrans_lmda[np.isneginf(Xrtrans_lmda)] = -1e+10

            Xbr_trans = np.einsum('npjk, nkr -> npjr', Xrtrans_lmda, Brtrans)  # (N, P, J, R)
            # Xbr_trans[np.isnan(Xbr_trans)] = 1e-30 # TODO
            # combining utilities
            V += Xbr_trans  # (N, P, J, R)

        #  Combine utilities of fixed and random variables
        V[V > 700] = 700
        # Exponent of the utility function for the logit formula
        eV = dev.np.exp(V)
        if avail is not None:
            eV = eV*avail[:, None, :, None]  # Acommodate availablity of alts.

        # Thresholds to avoid overflow warnings
        eV[np.isposinf(eV)] = 1e+30
        eV[np.isneginf(eV)] = -1e+30
        sum_eV = np.sum(eV, axis=2, keepdims=True)
        p = np.divide(eV, sum_eV, out=np.zeros_like(eV), where=(sum_eV != 0))
        p = p*panel_info[:, :, None, None]
        # Joint probability estimation for panels data
        pch = np.sum(y*p, axis=2) # (N, P, R)
        pch = self._prob_product_across_panels(pch, panel_info)
        # Thresholds to avoid divide by zero warnings
        pch[pch == 0] = 1e-300
        # Log-likelihood
        lik = pch.mean(axis=1)  # (N,)
        loglik = dev.np.log(lik)
        if weights is not None:
            loglik = loglik*weights
        loglik = loglik.sum()
        # Gradient estimation
        # Observed probability minus predicted probability
        ymp = y - p  # (N, P, J, R)
        # For fixed params
        # gradient = (Obs prob. minus predicted probability) * obs. var
        g = np.array([])
        if self.Kf != 0:
            g = np.einsum('npjr, npjk -> nkr', ymp, Xf)
            g = (g*pch[:, None, :]).mean(axis=2)/lik[:, None]
        # For random params w/ untransformed vars, two gradients will be
        # estimated: one for the mean and one for the s.d.
        # for mean: gr_b = (Obs. prob. minus pred. prob.)  * obs. var
        # for s.d.: gr_b = (Obs. prob. minus pred. prob.)  * obs. var * rand draw
        # if random coef. is lognormally dist:
        # gr_b = (obs. prob minus pred. prob.) * obs. var. * rand draw * der(R.V.)
        if self.Kr != 0:
            der = self._compute_derivatives(betas, draws, chol_mat=chol_mat)
            gr_b = np.einsum('npjr, npjk -> nkr', ymp, Xr)*der  # (N, Kr, R)
            # For correlation parameters
            # for s.d.: gr_w = (Obs prob. minus predicted probability) * obs. var * random draw
            draws_tril_idx = np.array([self.correlationpos[j] 
                                      for i in range(self.correlationLength)
                                       for j in range(i+1)])  # varnames pos.
            X_tril_idx = np.array([self.correlationpos[i]
                                   for i in range(self.correlationLength)
                                   for j in range(i+1)])
            # Find the s.d. for random variables that are not correlated
            range_var = [int(self.Kr - x - 1) for x in
                         list(range(self.correlationLength, self.Kr))]
            range_var = sorted(range_var)
            draws_tril_idx = np.array(np.concatenate((draws_tril_idx, range_var)))
            X_tril_idx = np.array(np.concatenate((X_tril_idx, range_var)))
            draws_tril_idx = draws_tril_idx.astype(int)
            X_tril_idx = X_tril_idx.astype(int)  # TODO: Check
            gr_w = gr_b[:, X_tril_idx, :]*draws[:, draws_tril_idx, :]  # (N,P,Kr,R)
            gr_b = (gr_b*pch[:, None, :]).mean(axis=2)/lik[:, None]  # (N,Kr)
            gr_w = (gr_w*pch[:, None, :]).mean(axis=2)/lik[:, None]  # (N,Kr)
            # Gradient for fixed and random params
            g = np.concatenate((g, gr_b, gr_w), axis=1) if g.size \
                else np.concatenate((gr_b, gr_w), axis=1)

        # For Box-Cox vars
        if len(self.transvars) > 0:
            if self.Kftrans:  # with fixed params
                gftrans = np.einsum('npjr, npjk -> nkr', ymp, Xftrans_lmda) # (N, Kf, R)
                # for the lambda param
                der_Xftrans_lmda = self.transform_deriv(Xftrans, flmbda)
                der_Xftrans_lmda[np.isposinf(der_Xftrans_lmda)] = 1e+30
                # der_Xftrans_lmda[np.isneginf(der_Xftrans_lmda)] = -1e+30
                der_Xftrans_lmda[np.isnan(der_Xftrans_lmda)] = 1e-30 # TODO 
                der_Xbftrans = np.einsum('npjk,k -> npk', der_Xftrans_lmda, Bftrans)
                gftrans_lmda = np.einsum('npjr,npk -> nkr', ymp, der_Xbftrans) # (N, Kfbc, R)
                gftrans = (gftrans*pch[:, None, :]).mean(axis=2)/lik[:, None]
                gftrans_lmda = (gftrans_lmda*pch[:, None, :]).mean(axis=2)/lik[:, None]
                
                g = np.concatenate((g, gftrans, gftrans_lmda), axis = 1) if g.size \
                    else np.concatenate((gftrans, gftrans_lmda), axis=1)
            if self.Krtrans:
                # for rand params
                # for mean: (obs prob. min pred. prob)*obs var * deriv rand coef
                # if rand coef is lognormally distributed:
                # gr_b = (obs prob minus pred. prob) * obs. var * rand draw * der(RV)
                #  TODO: Check
                temp_chol = chol_mat if chol_mat.size != 0 else np.diag(Brtrans_w)
                dertrans = self._compute_derivatives(Brtrans, draws=drawstrans, dist=self.rvtransdist, chol_mat=temp_chol, K=self.Krtrans)
                grtrans_b = np.einsum('npjr, npjk -> nkr', ymp, Xrtrans_lmda)*dertrans
                # for s.d. (obs - pred) * obs var * der rand coef * rand draw
                grtrans_w = np.einsum('npjr, npjk -> nkr', ymp, Xrtrans_lmda)*dertrans*drawstrans
                # for the lambda param
                # gradient = (obs - pred) * deriv x_lambda * beta
                der_Xrtrans_lmda = self.transform_deriv(Xrtrans, rlmda)
                der_Xrtrans_lmda[np.isposinf(der_Xrtrans_lmda)] = 1e+30
                der_Xrtrans_lmda[np.isnan(der_Xrtrans_lmda)] = 1e-30 # TODO 
                der_Xbrtrans = np.einsum('npjk, nkr -> npjkr', der_Xrtrans_lmda, Brtrans) # (N, P, J, K, R)
                grtrans_lmda = np.einsum('npjr, npjkr -> nkr', ymp, der_Xbrtrans) # (N, Krtrans, R)
                grtrans_b = (grtrans_b*pch[:, None, :]).mean(axis=2)/lik[:, None]  # (N,Kr)
                grtrans_w = (grtrans_w*pch[:, None, :]).mean(axis=2)/lik[:, None]  # (N,Kr)
                grtrans_lmda = (grtrans_lmda*pch[:, None, :]).mean(axis=2)/lik[:, None]  # (N,Kr)
                g = np.concatenate((g, grtrans_b, grtrans_w, grtrans_lmda), axis=1) if g.size \
                    else np.concatenate((grtrans_b, grtrans_w, grtrans_lmda), axis=1)

        # weighted average of the gradient when panels data is used
       
        # Hessian estimation
        H = g.T.dot(g)
        H[np.isnan(H)] = 1e-10 #TODO: why nan!!
        H[np.isposinf(H)] = 1e+30
        H[np.isneginf(H)] = -1e+30
        # H[H > 1e+10] = 1e+10
        # H[H < -1e+10] = -1e+10
        Hinv = np.linalg.pinv(H)
        self.total_fun_eval += 1

        # updated gradient
        if weights is not None:
            g = g*weights[:, None]
        g = np.sum(g, axis=0)  # (K, )
        if dev.using_gpu:
            g, loglik = dev.to_cpu(g), dev.to_cpu(loglik)
        # log-lik
        return -loglik, -g, Hinv

    # def _loglik_gradient(self, betas, X, y, panel_info, draws, drawstrans, weights):

    #     if dev.using_gpu:
    #         betas = dev.to_gpu(betas)
    #     p = self._compute_probabilities(betas, X, panel_info, draws, drawstrans, avail)
    #     # Probability of chosen alts
    #     pch = (y*p).sum(axis=2)  # (N,P,R)
    #     pch = self._prob_product_across_panels(pch, panel_info)  # (N,R)
    #     # Log-likelihood
    #     lik = pch.mean(axis=1)  # (N,)
    #     loglik = dev.np.log(lik)
    #     if weights is not None:
    #         loglik = loglik*weights
    #     loglik = loglik.sum()

    #     # Gradient
    #     Xf = X[:, :, :, ~self.rvidx]
    #     Xr = X[:, :, :, self.rvidx]

    #     ymp = y - p  # (N,P,J,R)
    #     # Gradient for fixed and random params
    #     gr_f = dev.np.einsum('npjr,npjk -> nkr', ymp, Xf)
    #     der = self._compute_derivatives(betas, draws)
    #     gr_b = dev.np.einsum('npjr,npjk -> nkr', ymp, Xr)*der
    #     gr_w = dev.np.einsum('npjr,npjk -> nkr', ymp, Xr)*der*draws
    #     # Multiply gradient by the chose prob. and dived by mean chose prob.
    #     gr_f = (gr_f*pch[:, None, :]).mean(axis=2)/lik[:, None]  # (N,Kf)
    #     gr_b = (gr_b*pch[:, None, :]).mean(axis=2)/lik[:, None]  # (N,Kr)
    #     gr_w = (gr_w*pch[:, None, :]).mean(axis=2)/lik[:, None]  # (N,Kr)
    #     # Put all gradients in a single array and aggregate them
    #     grad = self._concat_gradients(gr_f, gr_b, gr_w)  # (N,K)
    #     if weights is not None:
    #         grad = grad*weights[:, None]
    #     grad = grad.sum(axis=0)  # (K,)

    #     if dev.using_gpu:
    #         grad, loglik = dev.to_cpu(grad), dev.to_cpu(loglik)
    #     self.total_fun_eval += 1
    #     if self.verbose > 1:
    #         print("Evaluation {}  Log-Lik.={:.2f}".format(self.total_fun_eval,
    #                                                       -loglik))
    #     return -loglik, -grad

    def _concat_gradients(self, gr_f, gr_b, gr_w):
        idx = np.append(np.where(~self.rvidx)[0], np.where(self.rvidx)[0])
        gr_fb = np.concatenate((gr_f, gr_b), axis=1)[:, idx]
        return np.concatenate((gr_fb, gr_w), axis=1)

    def _prob_product_across_panels(self, pch, panel_info):
        if not np.all(panel_info):  # If panels unbalanced. Not all ones
            idx = panel_info == 0
            for i in range(pch.shape[2]):
                pch[:, :, i][idx] = 1  # Multiply by one when unbalanced
        pch = pch.prod(axis=1)  # (N,R)
        pch[pch == 0] = 1e-30
        return pch  # (N,R)

    def _apply_distribution(self, betas_random, index=None, draws=None):
        index = index if (index is not None) else self.rvdist
        for k, dist in enumerate(index):
            if dist == 'ln':
                betas_random[:, k, :] = dev.np.exp(betas_random[:, k, :])
            elif dist == 'tn':
                betas_random[:, k, :] = betas_random[:, k, :] *\
                    (betas_random[:, k, :] > 0)
        return betas_random

    def _balance_panels(self, X, y, panels):
        """Balance panels if necessary and produce a new version of X and y.

        If panels are already balanced, the same X and y are returned. This
        also returns panel_info, which keeps track of the panels that needed
        balancing.
        """
        _, J, K = X.shape
        _, p_obs = np.unique(panels, return_counts=True)
        p_obs = (p_obs/J).astype(int)
        N = len(p_obs)  # This is the new N after accounting for panels
        P = np.max(p_obs)  # panels length for all records
        if not np.all(p_obs[0] == p_obs):  # Balancing needed
            y = y.reshape(X.shape[0], J, 1)
            Xbal, ybal = np.zeros((N*P, J, K)), np.zeros((N*P, J, 1))
            panel_info = np.zeros((N, P))
            cum_p = 0  # Cumulative sum of n_obs at every iteration
            for n, p in enumerate(p_obs):
                # Copy data from original to balanced version
                Xbal[n*P:n*P + p, :, :] = X[cum_p:cum_p + p, :, :]
                ybal[n*P:n*P + p, :, :] = y[cum_p:cum_p + p, :, :]
                panel_info[n, :p] = np.ones(p)
                cum_p += p
        else:  # No balancing needed
            Xbal, ybal = X, y
            panel_info = np.ones((N, P))
        return Xbal, ybal, panel_info

    def _compute_derivatives(self, betas, draws, dist=None, K=None, chol_mat=None):
        N, R = draws.shape[0], draws.shape[2]
        Kr = K if K else self.Kr

        der = dev.np.ones((N, Kr, R))
        dist = dist if dist else self.rvdist
        if any(set(dist).intersection(['ln', 'tn'])):  # If any ln or tn
            _, betas_random = self._transform_betas(betas, draws, chol_mat=chol_mat)
            for k, dist in enumerate(dist):
                if dist == 'ln':
                    der[:, k, :] = betas_random[:, k, :]
                elif dist == 'tn':
                    der[:, k, :] = 1*(betas_random[:, k, :] > 0)
        return der

    def _transform_betas(self, betas, draws, trans=False, chol_mat=None):
        """Compute the products between the betas and the random coefficients.

        This method also applies the associated mixing distributions
        """
        # Extract coeffiecients from betas array
        # Kf = self.Kf # Number of fixed coeff
        betas_fixed = betas[0:self.Kf]  # First Kf positions
        br_mean = betas[self.Kf:self.Kf+self.Kr]
        # Compute: betas = mean + sd*draws
        # TODO: Consider randtrans?
        betas_random = br_mean[None, :, None] + np.matmul(chol_mat, draws)
        betas_random = self._apply_distribution(betas_random, self.rvdist, 
                                                draws=draws)
        return betas_fixed, betas_random

    def _generate_draws(self, sample_size, n_draws, halton=True):
        draws = drawstrans = []
        if halton:
            if self.randvars:
                draws = self._get_halton_draws(sample_size, n_draws,
                                               np.sum(self.rvidx))
            if self.randtransvars:
                drawstrans = self._get_halton_draws(sample_size, n_draws,
                                                    np.sum(self.rvtransidx))
        else:
            if self.randvars:
                draws = self._get_random_draws(sample_size, n_draws,
                                               np.sum(self.rvdist))
            if self.randtransvars:          
                drawstrans = self._get_random_draws(sample_size, n_draws,
                                                    np.sum(self.rvtransidx))
        # remove False to allow better enumeration
        self.rvdist = [x for x in self.rvdist if x is not False]
        for k, dist in enumerate(self.rvdist):
            if dist in ['n', 'ln', 'tn']:  # Normal based
                draws[:, k, :] = scipy.stats.norm.ppf(draws[:, k, :])
            elif dist == 't':  # Triangular
                draws_k = draws[:, k, :]
                draws[:, k, :] = (np.sqrt(2*draws_k) - 1)*(draws_k <= .5) +\
                    (1 - np.sqrt(2*(1 - draws_k)))*(draws_k > .5)
            elif dist == 'u':  # Uniform
                draws[:, k, :] = 2*draws[:, k, :] - 1

        self.rvtransdist = [x for x in self.rvtransdist if x is not False] #remove False to allow better enumeration
        for k, dist in enumerate(self.rvtransdist):
            if dist in ['n', 'ln', 'tn']:  # Normal based
                drawstrans[:, k, :] = scipy.stats.norm.ppf(drawstrans[:, k, :])
            elif dist == 't':  # Triangular
                draws_k = drawstrans[:, k, :]
                drawstrans[:, k, :] = (np.sqrt(2*drawstrans) - 1)*(drawstrans <= .5) +\
                    (1 - np.sqrt(2*(1 - drawstrans)))*(drawstrans > .5)
            elif dist == 'u':  # Uniform
                drawstrans[:, k, :] = 2*drawstrans[:, k, :] - 1
        return draws, drawstrans  # (N,Kr,R)

    def _get_random_draws(self, sample_size, n_draws, n_vars):
        return np.random.uniform(size=(sample_size, n_vars, n_draws))

    def _get_halton_draws(self, sample_size, n_draws, n_vars, shuffled=False):
        primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47,
                  53, 59, 61, 71, 73, 79, 83, 89, 97, 101, 103, 107,
                  109, 113, 127, 131, 137, 139, 149, 151, 157, 163, 167,
                  173, 179, 181, 191, 193, 197, 199]

        def halton_seq(length, prime=3, shuffled=False, drop=100):
            h = np.array([.0])
            t = 0
            while len(h) < length + drop:
                t += 1
                h = np.append(h, np.tile(h, prime-1) +
                              np.repeat(np.arange(1, prime)/prime**t, len(h)))
            seq = h[drop:length+drop]
            if shuffled:
                np.random.shuffle(seq)
            return seq
        draws = [halton_seq(sample_size*n_draws, prime=primes[i % len(primes)],
                            shuffled=shuffled).reshape(sample_size, n_draws)
                 for i in range(n_vars)]
        draws = np.stack(draws, axis=1)
        return draws

    def _model_specific_validations(self, randvars, Xnames):
        """Conduct validations specific for mixed logit models."""
        if randvars is None:
            raise ValueError("The randvars parameter is required for Mixed "
                             "Logit estimation")
        if not set(randvars.keys()).issubset(Xnames):
            raise ValueError("Some variable names in randvars were not found "
                             "in the list of variable names")
        if not set(randvars.values()).issubset(["n", "ln", "t", "tn", "u"]):
            raise ValueError("Wrong mixing distribution found in randvars. "
                             "Accepted distrubtions are n, ln, t, u, tn")

    def summary(self):
        """Show estimation results in console."""
        super(MixedLogit, self).summary()

    @staticmethod
    def check_if_gpu_available():
        """Check if GPU processing is available by running a quick estimation.

        Returns
        -------
        bool
            True if GPU processing is available, False otherwise.

        """
        n_gpus = dev.get_device_count()
        if n_gpus > 0:
            # Test a very simple example to see if CuPy is working
            X = np.array([[2, 1], [1, 3], [3, 1], [2, 4]])
            y = np.array([0, 1, 0, 1])
            model = MixedLogit()
            model.fit(X, y, varnames=["a", "b"], alts=["1", "2"], n_draws=500,
                      randvars={'a': 'n', 'b': 'n'}, maxiter=0, verbose=0)
            print("{} GPU device(s) available. xlogit will use "
                  "GPU processing".format(n_gpus))
            return True
        else:
            print("*** No GPU device found. Verify CuPy is properly installed")
            return False