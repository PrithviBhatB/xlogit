"""
Implements all the logic for mixed logit models
"""
# pylint: disable=invalid-name
from xlogit.boxcox_functions import boxcox_param_deriv_mixed, boxcox_transformation_mixed
import scipy.stats
from scipy.optimize import minimize
from ._choice_model import ChoiceModel
from ._device import device as dev
import numpy as np


class MixedLogit(ChoiceModel):
    """Class for estimation of Mixed Logit Models"""

    def __init__(self):
        """Init Function"""
        super(MixedLogit, self).__init__()
        self.rvidx = None  # Boolean index of random vars in X. True = rand var
        self.rvdist = None

    # X: (N, J, K)
    def fit(self, X, y, varnames=None, alt=None, isvars=None, transvars=None, id=None, transformation=None,
            weights=None, randvars=None, panel=None, base_alt=None,
            fit_intercept=False, init_coeff=None, maxiter=2000,
            random_state=None, n_draws=200, halton=True, verbose=1):

        X, y, initialData, varnames, alt, isvars, transvars, id, weights, panel\
            = self._as_array(X, y, varnames, alt, isvars, transvars, id, weights, panel)

        self._validate_inputs(X, y, alt, varnames, isvars, id, weights, panel,
                              base_alt, fit_intercept, maxiter)
        self._pre_fit(alt, varnames, isvars, transvars, base_alt,
                      fit_intercept, transformation, maxiter, randvars)

        if random_state is not None:
            np.random.seed(random_state)
        self.initialData = initialData

        X, y, panel = self._arrange_long_format(X, y, id, alt, panel)
        self.randvars = [x for x in randvars if x not in transvars]
        self.randtransvars = [x for x in transvars if x not in self.randvars]
        self.fixedtransvars = [x for x in transvars if x not in self.randtransvars]
        X, Xnames = self._setup_design_matrix(X)
        if self.transformation == "boxcox":
            self.transFunc = boxcox_transformation_mixed
            self.transform_deriv = boxcox_param_deriv_mixed
        N = self.N
        J = self.J
        K = (self.J-1) * len(self.isvars) if  len(self.isvars) > 0 else 2 #TODO: CHECK
        P = self.P
        R = n_draws
        # if panel is not None:  # If panel
        #     X, y, panel_info = self._balance_panels(X, y, panel)
        #     N, P = panel_info.shape
        # else:
        #     N, P = X.shape[0], 1
        #     panel_info = np.ones((
        # N, 1))
        panel_info = np.ones((N, 1))
        # X = X.reshape(N, P, J, K)
        y = y.reshape(N, P, J, 1)
        self.y = y
        self.n_draws = n_draws
        rvpos = [self.varnames.tolist().index(i) for i in self.randvars]
        randtranspos = [self.varnames.tolist().index(i) for i in self.randtransvars] # bc transformed variables with random coeffs
        self.rvidx = np.zeros(len(self.varnames), dtype=bool)
        self.rvidx[rvpos] = True  # True: Random var, False: Fixed var
        self.rvdist = list(self.randvars)
        self.rvtransidx = np.zeros(len(self.varnames), dtype=bool)
        self.rvtransidx[randtranspos] = True # True: Random var, False: Fixed var
        self.rvtransdist = list(self.randtransvars)
        self.verbose = verbose
        self.total_fun_eval = 0

        if weights is not None:
            weights = weights*(N/np.sum(weights))  # Normalize weights

        # Generate draws
        draws, drawstrans = self._generate_draws(N, R, halton)  # (N,Kr,R)
        n_coeff = self.Kf + 2*self.Kr + 2*self.Kftrans + 3*self.Krtrans
        if init_coeff is None:
            betas = np.repeat(.0, n_coeff)
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
        print('minimize')
        optimizat_res = \
            minimize(self._loglik_gradient_trans, betas, jac=True, method='BFGS',
                     args=(X, y, panel_info, draws, drawstrans, weights), tol=1e-4,
                     options={'gtol': 1e-4, 'maxiter': maxiter,
                              'disp': verbose > 0}
                              )
        print('optimizat_res', optimizat_res['x'])
        
        randpos = [Xnames.tolist().index(x) for x in Xnames if x in randvars]
        coeff_names = np.concatenate((Xnames,
                                      np.char.add("sd.", Xnames[randpos])))
        self._post_fit(optimizat_res, coeff_names, N, verbose)

    def _compute_probabilities(self, betas, X, panel_info, draws, drawstrans):
        Bf, Br = self._transform_betas(betas, draws)  # Get fixed and rand coef
        Xf = X[:, :, :, ~self.rvidx]  # Data for fixed coefficients
        self.Xr = X[:, :, :, self.rvidx]   # Data for random coefficients
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

    def _loglik_gradient_trans(self, betas, X, y, panel_info, draws, drawstrans, weights):

        if dev.using_gpu:
            betas = dev.to_gpu(betas)
        # Segregate initial values
        Bf = betas[0:self.Kf] # fixed
        Br_b = betas[self.Kf:self.Kf+self.Kr]
        Br_w = betas[self.Kf+self.Kr:self.Kf+2*self.Kr]
        Bftrans = betas[self.Kf+2*self.Kr:self.Kf+2*self.Kr+2*self.Kftrans]
        flmda = betas[self.Kf+2*self.Kr+self.Kftrans:self.Kf+2*self.Kr+2*self.Kftrans]
        Brtrans_b = betas[self.Kf+2*self.Kr+2*self.Kftrans:self.Kf+2*self.Kr+2*self.Kftrans+self.Krtrans]
        Brtrans_w = betas[self.Kf+2*self.Kr+2*self.Kftrans+self.Krtrans:self.Kf+2*self.Kr+2*self.Kftrans+2*self.Krtrans]
        rlmda = betas[self.Kf+2*self.Kr+2*self.Kftrans+2*self.Krtrans:]
        V = np.zeros((self.N, self.P, self.J, self.R))
        Xrtrans_lmda = None
        # Utility specification with the fixed coeffs
        if self.Kf != 0:
            self.Xbf = np.einsum('npjk,k -> npj', self.Xf, Bf)
            V += self.Xbf[:, :, :, None]*self.S[:, :, :, None]

        if self.Kr != 0:
            Br = Br_b[None, :, None] + draws*Br_w[None, :, None]
            Br = self._apply_distribution(Br, self.rvdist)
            Xbr = np.einsum('npjk, nkr -> npjr', self.Xr, Br)
            V += Xbr

        #transformation
        if (len(self.transvars) > 0):
            # transformations for variables with fixed coeffs
            if self.Kftrans != 0:
                Xftrans_lmda = self.transFunc(self.Xf_trans, flmda)
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
                Xbr_trans = np.einsum('npjk, nkr -> npjr', Xrtrans_lmda, Brtrans)
                # combining utilities
                V += Xbr_trans # (N, P, J, R)

        # thresholds to avoid overflow warnings
        V[V > 700] = 700
        # exponents of utilities for the logit formula
        eV = np.exp(V)

        # estimation of logit probabilities
        sumeV = np.sum(eV, axis=2, keepdims=True)
        sumeV[sumeV == 0] = 1e-30
        p = eV/sumeV

        # Joint probability estimation for panel data
        pch = np.sum(y*p, axis=2) # (N, P, R)
        pch = pch.prod(axis=1) # (N, R)
        pch[pch == 0] = 1e-30

        # Observed probability minus predicted probability
        ymp = y - p

        # For fixed params
        # gradient = (Observed prob minus predicted prob)*(obs variable)
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
            der = self._prithvi_compute_derivatives(Br, self.rvdist, self.Kr)
            gr_b = np.einsum('npjr, npjk -> nkr', ymp, self.Xr)*der # (N, Kr, R)
            gr_w = np.einsum('npjr, npjk -> nkr', ymp, self.Xr)*der*draws
            # Gradient for fixed and random oarams
            g = np.concatenate((g, gr_b, gr_w), axis = 1) if g.size else np.concatenate((gr_b, gr_w), axis = 1) 

        # For Box-Cox vars
        if len(self.transvars) > 0:
            if self.Kftrans:  # with fixed params
                gftrans = np.einsum('npjr, npjk -> nkr', ymp, Xftrans_lmda) # (N, Kf, R)

                # for the lambda param
                # gradient = (obs. prob - pred. prod) * transformed obs. var
                der_Xftrans_lmda = self.transform_deriv(self.Xf_trans, flmda)
                der_Xbftrans = np.einsum('npjk,k -> npj', der_Xftrans_lmda, Bftrans)
                gftrans_lmda = np.einsum('npjr, npjr -> npr', ymp, der_Xbftrans)

                g = np.concatenate((g, gftrans, gftrans_lmda), axis = 1) if g.size \
                    else np.concatenate((gftrans, gftrans_lmda), axis=1)
            if self.Krtrans:
                # for rand params
                # for mean: (obs prob. min pred. prob)*obs var * deriv rand coef
                # if rand coef is lognormally distributed:
                # gr_b = (obs prob minus pred. prob) * obs. var * rand draw * der(RV)
                dertrans = self._prithvi_compute_derivatives(Brtrans, self.rvtransdist, self.Krtrans)
                grtrans_b = np.einsum('npjr, npjk -> nkr', ymp, Xrtrans_lmda)*dertrans

                # for s.d. (obs - pred) * obs var * der rand coef * rand draw
                grtrans_w = np.einsum('npjr, npjk -> nkr', ymp, Xrtrans_lmda)*dertrans*drawstrans

                # for the lambda param
                # gradient = (obs - pred) * deriv x_lambda * beta
                der_Xrtrans_lmda = self.transform_deriv(self.Xr_trans, rlmda)
                der_Xbrtrans = np.einsum('npjk, nkr -> npjkr', der_Xrtrans_lmda, Brtrans) # (N, P, J, K, R)
                grtrans_lmda = np.einsum('npjr, npjkr -> nkr', ymp, der_Xbrtrans) # (N, Krtrans, R)
                g = np.concatenate((g, grtrans_b, grtrans_w, grtrans_lmda), axis = 1) if g.size \
                    else np.concatenate((grtrans_b, grtrans_w, grtrans_lmda), axis = 1)
        g = (g*pch[:, None, :]/np.mean(pch, axis=1)[:, None, None]) # (N, K, R)
        g = np.mean(g, axis=2) # (N, K)
        # Hessian estimation
        H = g.T.dot(g)
        # Hessian inverse
        Hinv = np.linalg.pinv(H)

        # Updated gradient
        g = np.sum(g, axis=0)
        self.total_fun_eval += 1

        # log lik
        l = np.mean(pch, axis=1)
        ll = np.sum(np.log(l))
        return (-ll, -g, Hinv)


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
        grad = dev.np.concatenate((gr_f, gr_b, gr_w), axis=1)  # (N,K)
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

    def _balance_panels(self, X, y, panel):
        _, J, K = X.shape
        _, p_obs = np.unique(panel, return_counts=True)
        p_obs = (p_obs/J).astype(int)
        N = len(p_obs)  # This is the new N after accounting for panels
        P = np.max(p_obs)  # Panel length for all records

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

    def _prithvi_compute_derivatives(self, Br, rpdist, Kr):
        der = np.ones((self.N, Kr, self.R))
        for k, dis in enumerate(rpdist):
            if dis=='ln':
                der[:, k, :] = Br[:, k, :]
        return der

    def _compute_derivatives(self, betas, draws):
        N, R, Kr = draws.shape[0], draws.shape[2], self.rvidx.sum()
        der = dev.np.ones((N, Kr, R))
        if any(set(self.rvdist).intersection(['ln', 'tn'])):  # If any ln or tn
            _, betas_random = self._transform_betas(betas, draws)
            for k, dist in enumerate(self.rvdist):
                if dist == 'ln':
                    der[:, k, :] = betas_random[:, k, :]
                elif dist == 'tn':
                    der[:, k, :] = 1*(betas_random[:, k, :] > 0)
        return der

    def _transform_betas(self, betas, draws):
        # Extract coeffiecients from betas array
        Kr = self.rvidx.sum()   # Number of random coeff
        Kf = len(betas) - 2*Kr  # Number of fixed coeff
        betas_fixed = betas[0:Kf]  # First Kf positions
        br_mean, br_sd = betas[Kf:Kf+Kr], betas[Kf+Kr:]  # Remaining positions
        # Compute: betas = mean + sd*draws
        betas_random = br_mean[None, :, None] + draws*br_sd[None, :, None]
        betas_random = self._apply_distribution(betas_random, draws)
        return betas_fixed, betas_random

    def _generate_draws(self, sample_size, n_draws, halton=True):
        if halton:
            draws = self._get_halton_draws(sample_size, n_draws,
                                           len(self.rvdist) + len(self.rvtransdist))
        else:
            draws = self._get_random_draws(sample_size, n_draws,
                                           len(self.rvdist + len(self.rvtransdist)))

        for k, dist in enumerate(self.rvdist):
            if dist in ['n', 'ln', 'tn']:  # Normal based
                draws[:, k, :] = scipy.stats.norm.ppf(draws[:, k, :])
            elif dist == 't':  # Triangular
                draws_k = draws[:, k, :]
                draws[:, k, :] = (np.sqrt(2*draws_k) - 1)*(draws_k <= .5) +\
                    (1 - np.sqrt(2*(1 - draws_k)))*(draws_k > .5)
            elif dist == 'u':  # Uniform
                draws[:, k, :] = 2*draws[:, k, :] - 1

        return draws  # (N,Kr,R)

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
        # draws = [halton_seq(sample_size*n_draws, prime=primes[i % len(primes)],
        #                     shuffled=shuffled).reshape(sample_size, n_draws)
        #          for i in range(n_vars)]
        # print('draws', draws)
        # draws = np.stack(draws, axis=1)
        def _getRandomDraws(sampleSize,numberOfDraws,symmetric=False,shuffled=False):
            normal_dist = scipy.stats.norm(loc=0.0, scale=1.0)
            numbers = normal_dist.rvs(size=(sampleSize, numberOfDraws))
            if shuffled:
                np.random.shuffle(numbers)
            return numbers
        draws = None
        drawstrans = None
        if self.randvars:
            draws = np.stack([_getRandomDraws(self.N, 100) for dis in self.rvdist], axis=1) #(N,Kr,R)
        if self.randtransvars:
            drawstrans = np.stack([_getRandomDraws(self.N, 100) for dis in self.rvtransdist], axis=1) #(N,Krbc,R)
        return draws, drawstrans
