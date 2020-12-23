"""
Implements all the logic for mixed logit models
"""
# pylint: disable=invalid-name
from numpy.lib.arraysetops import isin
from .boxcox_functions import boxcox_param_deriv_mixed, boxcox_transformation_mixed
import scipy.stats
from scipy.optimize import minimize, SR1
from ._choice_model import ChoiceModel
from ._device import device as dev
import numpy as np
import itertools


class MixedLogit(ChoiceModel):
    """Class for estimation of Mixed Logit Models"""

    def __init__(self):
        """Init Function"""
        super(MixedLogit, self).__init__()
        self.rvidx = None  # Boolean index of random vars in X. True = rand var
        self.rvdist = None

    # X: (N, J, K)
    def fit(self, X, y, varnames=None, alt=None, isvars=None, transvars=None, id=None, transformation=None,
            weights=None, randvars=None, mixby=None, panel=None, base_alt=None,
            fit_intercept=False, init_coeff=None, maxiter=2000,
            random_state=None, correlation=None, n_draws=200, halton=True, verbose=1, method="bfgs"):

        X, y, initialData, varnames, alt, isvars, transvars, id, weights, panel\
            = self._as_array(X, y, varnames, alt, isvars, transvars, id, weights, panel)

        self._validate_inputs(X, y, alt, varnames, isvars, id, weights, panel,
                              base_alt, fit_intercept, maxiter)
        self._pre_fit(alt, varnames, isvars, transvars, base_alt,
                      fit_intercept, transformation, maxiter, panel, correlation, randvars)
        rvpos = [self.varnames.index(i) for i in self.randvars]
        randtranspos = [self.varnames.__add__index(i) for i in self.randtransvars] # bc transformed variables with random coeffs
        self.rvidx, self.rvdist = [], []
        self.randvars = [x for x in self.randvars if x not in transvars] # random variables not transformed
        self.randtransvars = [x for x in transvars if (x in randvars) and (x not in self.randvars)]
        self.fixedtransvars = [x for x in transvars if x not in self.randtransvars]
        self.n_draws = n_draws
        for var in self.varnames:
            if var in self.randvars:
                self.rvidx.append(True)
                self.rvdist.append(var)
            else:
                self.rvidx.append(False)
        self.rvtransidx, self.rvtransdist = [], []
        for var in self.varnames:
            if var in self.randtransvars:
                self.rvtransidx.append(True)
                self.rvtransdist.append(var)
            else:
                self.rvtransidx.append(False)


        if random_state is not None:
            np.random.seed(random_state)
        self.initialData = initialData
        X, y, panel = self._arrange_long_format(X, y, id, alt, panel)
        self.y = y
        X, Xnames = self._setup_design_matrix(X)
        # self.K = self.Xf.shape[2]
        if self.transformation == "boxcox":
            self.transFunc = boxcox_transformation_mixed
            self.transform_deriv = boxcox_param_deriv_mixed
        N = self.N
        P = self.P
        R = n_draws
        # panel_info = np.ones((N, 1))

        self.n_draws = n_draws
        self.n_draws = n_draws
        self.verbose = verbose
        self.total_fun_eval = 0

        if weights is not None:
            weights = weights*(N/np.sum(weights))  # Normalize weights

        # Generate draws
        draws, drawstrans = self._generate_draws(self.N, R, halton)  # (N,Kr,R)
        n_coeff = self.Kf + 2*self.Kr + 2*self.Kftrans + 3*self.Krtrans
        if init_coeff is None:
            betas = np.concatenate((np.repeat(-0.1, self.Kf), np.repeat(0.2, self.Kr),
                     np.repeat(0.2, self.Kchol), np.repeat(0.3, self.Kbw), np.repeat(0.2, 2*self.Kftrans), np.repeat(0.2, 2*self.Krtrans), np.repeat(1, self.Krtrans)))
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
            if verbose > 0:
                print("Estimation with GPU processing enabled.")
        positive_bound = (0, 1e+30)
        any_bound = (-1e+30, 1e+30)
        corr_bound = (-1, 1)

        bound_dict = {
            "bf": (any_bound, self.Kf),
            "br_b": (any_bound, self.Kr),
            "chol": (any_bound, self.Kchol),
            "br_w": (positive_bound, self.Kr - self.correlationLength),
            "bf_trans": (any_bound, self.Kftrans),
            "flmbda": (any_bound, self.Kftrans),
            "br_trans_b": (any_bound, self.Krtrans),
            "br_trans_w": (positive_bound, self.Krtrans),
            "rlmbda": (any_bound, self.Krtrans)
        }

        # list comrephension to add number of bounds for each variable type
        bnds = [ (bound[1][0],) * int(bound[1][1]) for bound in bound_dict.items() if bound[1][1] > 0 ]
        bnds = [tuple(itertools.chain.from_iterable(bnds))][0]

        optimizat_res = \
            minimize(self._loglik_gradient_corr, betas, jac=True, method=method,
                     args=(X, y, self.panel_info, draws, drawstrans, weights), tol=1e-8,
                     bounds=bnds,
                     options={'gtol': 1e-8, 'maxiter': maxiter,
                              'disp': verbose > 0}
                              )
        self._post_fit(optimizat_res, Xnames, N, verbose)

    def _compute_probabilities(self, betas, X, panel_info, draws, drawstrans):
        Bf, Br = self._transform_betas(betas, draws)  # Get fixed and rand coef
        X = X.reshape((self.N, self.P, self.J, self.R)) # TODO: 6??? R = 100 ?? What?
        Xf = X[:, :, :, ~self.rvidx]  # Data for fixed coefficients
        self.Xr = X[:, :, :, self.rvidx]   # Data for random coefficients
        V = np.zeros_like(X, dtype=float)
        if (len(Bf) > 0):
            XBf = dev.np.einsum('npjk,k -> npj', Xf, Bf)  # (N,P,J)
            V = XBf[:, :, :, None]
        if (len(Br) > 1):
            XBr = dev.np.einsum('npjk,nkr -> npjr', self.Xr, Br)  # (N,P,J,R)
            V += XBr  # (N,P,J,R)
        V[V > 700] = 700
        eV = dev.np.exp(V)
        sumeV = dev.np.sum(eV, axis=2, keepdims=True)
        sumeV[sumeV == 0] = 1e-30
        p = eV/sumeV  # (N,P,J,R)
        p = p*panel_info[:, :, None, None]  # Zero for unbalanced panels
        return p

    def _loglik_gradient_corr(self, betas, X, y, panel_info, draws, drawstrans, weights):
        # Segregating initial values to fixed betas (Bf) and random beta means (Br_b)
        # for both non-transformed and transformed variables
        # and random beta cholesky factors (chol)
    
        if dev.using_gpu:
            betas = dev.to_gpu(betas)
        print('betasdebug', betas, len(betas))

        beta_segment_names = ["Bf", "Br_b", "chol", "Br_w", "Bftrans", "flmbda",
                              "Brtrans_b", "Brtrans_w", "rlmda"]
        var_list = dict()
        iterations = [self.Kf, self.Kr, self.Kchol, self.Kbw, self.Kftrans,
                      self.Kftrans, self.Krtrans, self.Krtrans, self.Krtrans]
        i = 0
        for count, iteration in enumerate(iterations):
            prev_index = i
            i = int(i + iteration)
            var_list[beta_segment_names[count]] = betas[prev_index:i]
        
        Bf, Br_b, chol, Br_w, Bftrans, flmbda, Brtrans_b, Brtrans_w, rlmda  = var_list.values()

        chol_mat = np.zeros((self.correlationLength, self.correlationLength))
        indices = np.tril_indices(self.correlationLength)
        chol_mat[indices] = chol
        chol_mat_temp = np.zeros((self.Kr, self.Kr))
        chol_mat_temp[:self.correlationLength, :self.correlationLength] = chol_mat

        for i in range(self.Kr - self.correlationLength):
            chol_mat_temp[i+self.correlationLength, i+self.correlationLength] = \
                Br_w[i]
        chol_mat = chol_mat_temp

        # Creating random coeffs using Br_b, cholesky matrix and random draws
        # Estimating the linear utility specification (U = sum of Xb)
        V = np.zeros((self.N, self.P, self.J, self.n_draws))

        if self.Kf != 0:
            XBf = np.einsum('npjk,k -> npj', self.Xf, Bf)
            V += XBf[:, :, :, None]*self.S[:, :, :, None]
        if self.Kr != 0:
            Br = Br_b[None, :, None] + np.matmul(chol_mat, draws)
            Br = self._apply_distribution(Br, self.rvdist)
            XBr = np.einsum('npjk, nkr -> npjr', self.Xr, Br) # (N, P, J, R)
            V += XBr*self.S[:, :, :, None]

        #  transformation
        if (len(self.transvars) > 0):
            #  transformations for variables with fixed coeffs
            if self.Kftrans != 0:
                Xftrans_lmda = self.transFunc(self.Xf_trans, flmbda)
                Xftrans_lmda[np.isneginf(Xftrans_lmda)] = 1e-5
                # Estimating the linear utility specificiation (U = sum XB)
                Xbf_trans = np.einsum('npjk,k -> npj', Xftrans_lmda, Bftrans)
                # combining utilities
                V += Xbf_trans[:, :, :, None]

        # transformations for variables with random coeffs
        if self.Krtrans != 0:
            # creating the random coeffs
            Brtrans = Brtrans_b[None, :, None] + drawstrans[:, 0:self.Krtrans, :] * Brtrans_w[None, :, None]  # TODO: draws BC!
            Brtrans = self._apply_distribution(Brtrans, self.rvtransdist)
            # applying transformation 
            Xrtrans_lmda = self.transFunc(self.Xr_trans, rlmda)
            Xrtrans_lmda[np.isneginf(Xrtrans_lmda)] = -1e+10
            Xbr_trans = np.einsum('npjk, nkr -> npjr', Xrtrans_lmda, Brtrans)
            # combining utilities
            V += Xbr_trans # (N, P, J, R)

        #  Combine utilities of fixed and random variables
        V[V > 700] = 700
        # Exponent of the utility function for the logit formula
        eV = np.exp(V)

        # Thresholds to avoid overflow warnings
        eV[np.isposinf(eV)] = 1e+30
        eV[np.isneginf(eV)] = 1e-30
        sum_eV = np.sum(eV, axis=2, keepdims=True)
        p = np.divide(eV, sum_eV, out=np.zeros_like(eV), where=(sum_eV != 0))
        p = p*self.panel_info[:, :, None, None]
        # Joint probability estimation for panel data
        pch = np.sum(self.y*p, axis=2) # (N, P, R)
        # pch = pch.prod(axis=1) # (N, R)
        pch = self._prob_product_across_panels(pch, self.panel_info)
        # Thresholds to avoid divide by zero warnings
        pch[pch == 0] = 1e-300

        # Gradient estimation
        # Observed probability minus predicted probability
        ymp = self.y - p # (N, P, J, R)
        # For fixed params
        # gradient = (Obs prob. minus predicted probability) * obs. var
        g = np.array([])
        if self.Kf != 0:
            g = np.einsum('npjr, npjk -> nkr', ymp, self.Xf)
        # For random params w/ untransformed vars, two gradients will be
        # estimated: one for the mean and one for the s.d.
        # for mean: gr_b = (Obs. prob. minus pred. prob.)  * obs. var
        # for s.d.: gr_b = (Obs. prob. minus pred. prob.)  * obs. var * rand draw
        # if random coef. is lognormally dist:
        # gr_b = (obs. prob minus pred. prob.) * obs. var. * rand draw * der(R.V.)
        if self.Kr != 0:
            der = self._compute_derivatives(Br, draws, K=self.Kr)
            gr_b = np.einsum('npjr, npjk -> nkr', ymp, self.Xr)*der # (N, Kr, R)
            # For correlation parameters
            # for s.d.: gr_w = (Obs prob. minus predicted probability) * obs. var * random draw
            draws_tril_idx = np.array([self.correlationpos[j] for i in range(self.correlationLength) for j in range(i+1)]) # position in varnames
            X_tril_idx = np.array([self.correlationpos[i] for i in range(self.correlationLength) for j in range(i+1)])
            # Find the standard deviation for random variables that a not correlated
            range_var = [int(self.Kr - x - 1) for x in list(range(self.correlationLength, self.Kr))]# indices for s.d. of uncorrelated variables
            range_var = sorted(range_var)
            draws_tril_idx = np.array(np.concatenate((draws_tril_idx, range_var)))
            X_tril_idx = np.array(np.concatenate((X_tril_idx, range_var)))
            draws_tril_idx = draws_tril_idx.astype(int)
            X_tril_idx = X_tril_idx.astype(int)
            gr_w = gr_b[:, X_tril_idx, :]*draws[:, draws_tril_idx, :] # (N, P, Kr, R)
            # Gradient for fixed and random oarams
            g = np.concatenate((g, gr_b, gr_w), axis = 1) if g.size else np.concatenate((gr_b, gr_w), axis = 1) 

        # For Box-Cox vars
        if len(self.transvars) > 0:
            if self.Kftrans:  # with fixed params
                gftrans = np.einsum('npjr, npjk -> nkr', ymp, Xftrans_lmda) # (N, Kf, R)

                # for the lambda param
                # gradient = (obs. prob - pred. prod) * transformed obs. var
                der_Xftrans_lmda = self.transform_deriv(self.Xf_trans, flmbda)
                der_Xftrans_lmda[np.isposinf(der_Xftrans_lmda)] = 1e+30
                der_Xftrans_lmda[np.isneginf(der_Xftrans_lmda)] = 1e-30
                der_Xftrans_lmda[np.isnan(der_Xftrans_lmda)] = 1e-30 # TODO 
                der_Xbftrans = np.einsum('npjk,k -> npj', der_Xftrans_lmda, Bftrans)
                gftrans_lmda = np.einsum('npjr, npjr -> npr', ymp, der_Xbftrans)

                g = np.concatenate((g, gftrans, gftrans_lmda), axis = 1) if g.size \
                    else np.concatenate((gftrans, gftrans_lmda), axis=1)
            if self.Krtrans:
                # for rand params
                # for mean: (obs prob. min pred. prob)*obs var * deriv rand coef
                # if rand coef is lognormally distributed:
                # gr_b = (obs prob minus pred. prob) * obs. var * rand draw * der(RV)
                dertrans = self._compute_derivatives(Brtrans, drawstrans, dist=self.rvtransdist, K=self.Krtrans)
                grtrans_b = np.einsum('npjr, npjk -> nkr', ymp, Xrtrans_lmda)*dertrans

                # for s.d. (obs - pred) * obs var * der rand coef * rand draw
                grtrans_w = np.einsum('npjr, npjk -> nkr', ymp, Xrtrans_lmda)*dertrans*drawstrans

                # for the lambda param
                # gradient = (obs - pred) * deriv x_lambda * beta
                der_Xrtrans_lmda = self.transform_deriv(self.Xr_trans, rlmda)
                der_Xrtrans_lmda[np.isposinf(der_Xrtrans_lmda)] = 1e+30
                der_Xrtrans_lmda[np.isnan(der_Xrtrans_lmda)] = 1e-30 # TODO 
                der_Xbrtrans = np.einsum('npjk, nkr -> npjkr', der_Xrtrans_lmda, Brtrans) # (N, P, J, K, R)
                grtrans_lmda = np.einsum('npjr, npjkr -> nkr', ymp, der_Xbrtrans) # (N, Krtrans, R)
                g = np.concatenate((g, grtrans_b, grtrans_w, grtrans_lmda), axis = 1) if g.size \
                    else np.concatenate((grtrans_b, grtrans_w, grtrans_lmda), axis = 1)

        # weighted average of the gradient when panel data is used
        g = (g*pch[:, None, :]) / np.mean(pch, axis=1)[:, None, None] # (N, K, R)
        g = np.mean(g, axis=2) # (N, K)
        # Hessian estimation
        H = g.T.dot(g)
        H[np.isnan(H)] = 1e-10 #TODO: FIX!!
        H[np.isposinf(H)] = 1e+10
        H[np.isneginf(H)] = -1e+10
        H[H > 1e+10] = 1e+10
        H[H < -1e+10] = -1e+10
        Hinv = np.linalg.pinv(H)
        self.total_fun_eval += 1

        # updated gradient
        g = np.sum(g, axis=0) # (K, )
        # log-lik
        l = np.mean(pch, axis=1)
        ll = np.sum(np.log(l))
        return -ll, -g, Hinv


    def _loglik_gradient(self, betas, X, y, panel_info, draws, drawstrans, weights):

        if dev.using_gpu:
            betas = dev.to_gpu(betas)
        p = self._compute_probabilities(betas, X, panel_info, draws, drawstrans)
        # Probability of chosen alt
        pch = (y*p).sum(axis=2)  # (N,P,R)
        pch = self._prob_product_across_panels(pch, panel_info)  # (N,R)

        # Log-likelihood
        lik = pch.mean(axis=1)  # (N,)
        loglik = dev.np.log(lik)
        if weights is not None:
            loglik = loglik*weights
        loglik = loglik.sum()

        # Gradient
        Xf = X[:, :, :, ~self.rvidx]
        Xr = X[:, :, :, self.rvidx]

        ymp = y - p  # (N,P,J,R)
        # Gradient for fixed and random params
        gr_f = dev.np.einsum('npjr,npjk -> nkr', ymp, Xf)
        der = self._compute_derivatives(betas, draws)
        gr_b = dev.np.einsum('npjr,npjk -> nkr', ymp, Xr)*der
        gr_w = dev.np.einsum('npjr,npjk -> nkr', ymp, Xr)*der*draws
        # Multiply gradient by the chose prob. and dived by mean chose prob.
        gr_f = (gr_f*pch[:, None, :]).mean(axis=2)/lik[:, None]  # (N,Kf)
        gr_b = (gr_b*pch[:, None, :]).mean(axis=2)/lik[:, None]  # (N,Kr)
        gr_w = (gr_w*pch[:, None, :]).mean(axis=2)/lik[:, None]  # (N,Kr)
        # Put all gradients in a single array and aggregate them
        grad = self._concat_gradients(gr_f, gr_b, gr_w)  # (N,K)
        if weights is not None:
            grad = grad*weights[:, None]
        grad = grad.sum(axis=0)  # (K,)

        if dev.using_gpu:
            grad, loglik = dev.to_cpu(grad), dev.to_cpu(loglik)
        self.total_fun_eval += 1
        if self.verbose > 1:
            print("Evaluation {}  Log-Lik.={:.2f}".format(self.total_fun_eval,
                                                          -loglik))
        return -loglik, -grad

    def _concat_gradients(self, gr_f, gr_b, gr_w):
        idx = np.append(np.where(~self.rvidx)[0], np.where(self.rvidx)[0])
        gr_fb = np.concatenate((gr_f, gr_b), axis=1)[:, idx]
        return np.concatenate((gr_fb, gr_w), axis=1)

    def _prob_product_across_panels(self, pch, panel_info):
        if not np.all(panel_info):  # If panel unbalanced. Not all ones
            idx = panel_info == 0
            for i in range(pch.shape[2]):
                pch[:, :, i][idx] = 1  # Multiply by one when unbalanced
        pch = pch.prod(axis=1)  # (N,R)
        pch[pch == 0] = 1e-30
        return pch  # (N,R)

    def _apply_distribution(self, betas_random, draws):
        for k, dist in enumerate(self.rvdist):
            if dist == 'ln':
                betas_random[:, k, :] = dev.np.exp(betas_random[:, k, :])
            elif dist == 'tn':
                betas_random[:, k, :] = betas_random[:, k, :] *\
                    (betas_random[:, k, :] > 0)
        return betas_random

    # def _balance_panels(self, X, y, panel):
    #     X = self.Xf
    #     temp_dim, J, K, num_fixed_coeffs = X.shape[0], X.shape[1], X.shape[2], X.shape[3]
    #     _, p_obs = np.unique(panel, return_counts=True)
    #     p_obs = (p_obs/J).astype(int)
    #     N = len(p_obs)  # This is the new N after accounting for panels
    #     P = np.max(p_obs)  # Panel length for all records
    #     if not np.all(p_obs[0] == p_obs):  # Balancing needed
    #         y = y.reshape(X.shape[0], self.J, 1)
    #         Xbal = np.zeros((N*P, J, K, num_fixed_coeffs))
    #         ybal = np.zeros((N*P, self.J, 1))
    #         panel_info = np.zeros((N, P))
    #         cum_p = 0  # Cumulative sum of n_obs at every iteration
    #         for n, p in enumerate(p_obs):
    #             # Copy data from original to balanced version
    #             Xbal[n*P:n*P + p, :, :, :] = X[cum_p:cum_p + p, :, :, :]
    #             ybal[n*P:n*P + p, :, :] = y[cum_p:cum_p + p, :, :]
    #             panel_info[n, :p] = np.ones(p)
    #             cum_p += p

    #     else:  # No balancing needed
    #         Xbal, ybal = X, y
    #         panel_info = np.ones((N, P))
    #     return Xbal, ybal, panel_info

    # def _prithvi_compute_derivatives(self, Br, rpdist, Kr):
    #     der = np.ones((self.N, Kr, self.n_draws))
    #     for k, dis in enumerate(rpdist):
    #         if dis=='ln':
    #             der[:, k, :] = Br[:, k, :]
    #     return der

    def _compute_derivatives(self, betas, draws, dist=None, K=None):
        N, R = draws.shape[0], draws.shape[2]
        Kr = K if K else self.Kr
        der = dev.np.ones((N, Kr, R))
        dist = dist if dist else self.rvdist
        if any(set(dist).intersection(['ln', 'tn'])):  # If any ln or tn
            _, betas_random = self._transform_betas(betas, draws)
            for k, dist in enumerate(dist):
                if dist == 'ln':
                    der[:, k, :] = betas_random[:, k, :]
                elif dist == 'tn':
                    der[:, k, :] = 1*(betas_random[:, k, :] > 0)
        return der

    def _transform_betas(self, betas, draw, trans=False):
        # Extract coeffiecients from betas array
        Kf = len(betas) - 2*Kr  # Number of fixed coeff
        betas_fixed = betas[0:Kf]  # First Kf positions
        br_mean, br_sd = betas[Kf:Kf+Kr], betas[Kf+Kr:]  # Remaining positions
        # Compute: betas = mean + sd*draws
        betas_random = br_mean[None, :, None] + draws*br_sd[None, :, None]
        betas_random = self._apply_distribution(betas_random, draws)
        return betas_fixed, betas_random

    def _generate_draws(self, sample_size, n_draws, halton=True):
        draws = drawstrans = []
        if halton:
            if self.randvars:
                draws = self._get_halton_draws(sample_size, n_draws,
                                           len(self.rvdist))
            if self.randtransvars:
                drawstrans = self._get_halton_draws(sample_size, n_draws,
                                           len(self.rvtransdist))
        else:
            if self.randvars:
                draws = self._get_random_draws(sample_size, n_draws,
                                           len(self.rvdist))
            if self.randtransvars:          
                drawstrans = self._get_random_draws(sample_size, n_draws,
                                            len(self.rvtransdist))

        for k, dist in enumerate(self.rvdist):
            if dist in ['n', 'ln', 'tn']:  # Normal based
                draws[:, k, :] = scipy.stats.norm.ppf(draws[:, k, :])
            elif dist == 't':  # Triangular
                draws_k = draws[:, k, :]
                draws[:, k, :] = (np.sqrt(2*draws_k) - 1)*(draws_k <= .5) +\
                    (1 - np.sqrt(2*(1 - draws_k)))*(draws_k > .5)
            elif dist == 'u':  # Uniform
                draws[:, k, :] = 2*draws[:, k, :] - 1

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
        # normal_dist = scipy.stats.norm(loc=1.0, scale=0)
        # numbers = normal_dist.rvs(size=(sample_size, n_draws))
        # if shuffled:
        #     np.random.shuffle(numbers)
        # return numbers
    
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

    @staticmethod
    def check_if_gpu_available():
        X = np.array([[2, 1], [1, 3], [3, 1], [2, 4]])
        y = np.array([0, 1, 0, 1])
        model = MixedLogit()
        model.fit(X, y, varnames=["a", "b"], alt=["1", "2"], n_draws=500,
                  randvars={'a': 'n', 'b': 'n'}, maxiter=0, verbose=0)