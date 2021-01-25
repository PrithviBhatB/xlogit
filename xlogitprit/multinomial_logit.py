"""
Implements multinomial and conditional logit models
"""
# pylint: disable=line-too-long,invalid-name

import numpy as np
from ._choice_model import ChoiceModel
from .boxcox_functions import boxcox_transformation, boxcox_param_deriv
from scipy.optimize import minimize


class MultinomialLogit(ChoiceModel):
    """Class for estimation of Multinomial and Conditional Logit Models"""

    def fit(self, X, y, varnames=None, alts=None, isvars=None, transvars=None,
            transformation=None, ids=None, weights=None, avail=None,
            base_alt=None, fit_intercept=False, init_coeff=None, maxiter=2000,
            random_state=None, verbose=1):
        """Fit multinomial and/or conditional logit models.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_variables)
            Input data for explanatory variables in long format

        y : array-like, shape (n_samples,)
            Choices in long format

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

        verbose : int, default=1
            Verbosity of messages to show during estimation. 0: No messages,
            1: Some messages, 2: All messages


        Returns
        -------
        None.
        """
        X, y, initialData, varnames, alts, isvars, transvars, ids, weights, panels, panel \
            = self._as_array(X, y, varnames, alts, isvars, transvars, ids, weights, None, avail)
        self._validate_inputs(X, y, alts, varnames, isvars, ids, weights, None,
                              base_alt, fit_intercept, maxiter)

        self._pre_fit(alts, varnames, isvars, transvars, base_alt,
                      fit_intercept, transformation, maxiter, panels)
        self.fxidx, self.fxtransidx = [], []
        for var in self.varnames:
            if isvars is not None:
                if var in isvars:
                    continue
            if var in transvars:
                self.fxidx.append(False)
                self.fxtransidx.append(True)
            else:
                self.fxtransidx.append(False)
                self.fxidx.append(True)

        X, y, panels = self._arrange_long_format(X, y, ids, alts)
        self.y = y
        self.initialData = initialData
        if random_state is not None:
            np.random.seed(random_state)

        if self.transvars is not None and self.transformation is None:
            # if transvars provided and no specified transformation function
            # give default to boxcox
            self.transformation = "boxcox"

        if transformation == "boxcox":
            self.transFunc = boxcox_transformation
            self.transform_deriv = boxcox_param_deriv

        if avail is not None:
            avail = avail.reshape(X.shape[0], X.shape[1])

        if init_coeff is None:
            betas = np.repeat(.0, self.numFixedCoeffs + self.numTransformedCoeffs)
        else:
            betas = init_coeff
            if len(init_coeff) != (self.numFixedCoeffs + self.numTransformedCoeffs):
                raise ValueError("The size of initial_coeff must be: "
                                 + int(X.shape[1]))

        X, Xnames = self._setup_design_matrix(X)

        # add transformation vars and corresponding lambdas
        lambda_names = ["lambda.{}".format(transvar) for transvar in transvars]
        transnames = np.concatenate((transvars, lambda_names))
        Xnames = np.concatenate((Xnames, transnames))
        y = y.reshape(self.N, self.J)

        # Call optimization routine
        optimizat_res = self._scipy_bfgs_optimization(betas, X, y, weights,
                                                      avail, maxiter)
        self._post_fit(optimizat_res, Xnames, int(1182/4), verbose)

    def _compute_probabilities(self, betas, X, avail):
        Xf = X[:, :, self.fxidx]
        X_trans = X[:, :, self.fxtransidx]
        XB = 0
        if self.numFixedCoeffs > 0:
            B = betas[0:self.Kf]
            XB = Xf.dot(B)
        Xtrans_lambda = None
        if sum(self.fxtransidx):
            B_transvars = betas[self.numFixedCoeffs:int(self.numFixedCoeffs+(self.numTransformedCoeffs/2))]
            lambdas = betas[int(self.numFixedCoeffs+(self.numTransformedCoeffs/2)):]
            # applying transformations
            Xtrans_lambda = self.transFunc(X_trans, lambdas)
            XB_trans = Xtrans_lambda.dot(B_transvars)
            XB += XB_trans
        # XB[np.isnan(XB)] = 1e-30
        XB[XB > 700] = 700  # avoiding infs
        # XB[np.isposinf(XB)] = 1e+30  # avoiding infs
        # XB[np.isneginf(XB)] = -1e+30  # avoiding infs
        XB = XB.reshape(self.N, self.J)
        eXB = np.exp(XB)
        if avail is not None:
            eXB = eXB*avail
        eXB[np.isposinf(eXB)] = 1e+30
        eXB[np.isneginf(eXB)] = 1e-30
        p = eXB/np.sum(eXB, axis=1, keepdims=True)  # (N,J)
        return p, Xtrans_lambda

    def _loglik_and_gradient(self, betas, X, y, weights, avail):
        p, Xtrans_lmda = self._compute_probabilities(betas, X, avail)
        # Log likelihood
        lik = np.sum(y*p, axis=1)
        loglik = np.log(lik)
        if weights is not None:
            loglik = loglik*weights
        loglik = np.sum(loglik)
        # Individual contribution to the gradient

        transpos = [self.varnames.tolist().index(i) for i in self.transvars]  # Position of trans vars
        B_trans = betas[self.numFixedCoeffs:
                        int(self.numFixedCoeffs+(self.numTransformedCoeffs/2))]
        lambdas = betas[int(self.numFixedCoeffs+(self.numTransformedCoeffs/2)):]
        X_trans = self.initialData[:, transpos]
        X_trans = X_trans.reshape(self.N, len(self.alternatives), len(transpos))
        ymp = y - p
        if self.Kf > 0:
            Xf = X[:, :, self.fxidx]
            grad = np.einsum('nj,njk -> nk', ymp, Xf)
        else:
            grad = np.array([])
        if self.numTransformedCoeffs > 0:
            # Individual contribution of trans to the gradient
            gtrans = np.einsum('nj,njk -> nk', ymp, Xtrans_lmda)
            der_Xtrans_lmda = self.transform_deriv(X_trans, lambdas)
            der_XBtrans = np.einsum('njk,k -> njk', der_Xtrans_lmda, B_trans)
            gtrans_lmda = np.einsum('nj,njk -> nk', ymp, der_XBtrans)
            grad = np.concatenate((grad, gtrans, gtrans_lmda), axis=1) \
                if grad.size \
                else np.concatenate((gtrans, gtrans_lmda), axis=1)  # (N, K)
        if weights is not None:
            grad = grad*weights[:, None]

        H = np.dot(grad.T, grad)
        # H[H == 0] = 1e-10
        # H[np.isnan(H)] = 1e-10  # TODO: why nans!
        # H[H > 1e+10] = 1e+10
        # H[H < -1e+10] = -1e+10
        Hinv = np.linalg.pinv(H)

        grad = np.sum(grad, axis=0)
        return (-loglik, -grad, Hinv)

    def _scipy_bfgs_optimization(self, betas, X, y, weights, avail, maxiter):
        # res_init, g = self._loglik_and_gradient(betas, X, y, weights, avail)
        res = minimize(self._loglik_and_gradient,
                       betas,
                       args=(X, y, weights, avail),
                       jac=True,
                       method='BFGS',
                       tol=1e-10,
                       options={'gtol': 1e-10}
                       )
        return res

    def _bfgs_optimization(self, betas, X, y, weights, avail, maxiter):
        res, g, Hinv = self._loglik_and_gradient(betas, X, y, weights, avail)
        current_iteration = 0
        convergence = False
        while True:
            old_g = g
            d = -Hinv.dot(g)
            step = 2
            while True:
                step = step/2
                s = step*d
                betas = betas + s
                resnew, gnew, _ = self._loglik_and_gradient(betas, X, y,
                                                            weights, avail)
                if resnew <= res or step < 1e-10:
                    break

            old_res = res
            res = resnew
            g = gnew
            delta_g = g - old_g

            Hinv = Hinv + (((s.dot(delta_g) + (delta_g[None, :].dot(Hinv)).dot(
                delta_g))*np.outer(s, s)) / (s.dot(delta_g))**2) - ((np.outer(
                    Hinv.dot(delta_g), s) + (np.outer(s, delta_g)).dot(Hinv)) /
                    (s.dot(delta_g)))
            current_iteration = current_iteration + 1
            if np.abs(res - old_res) < 0.00001:
                convergence = True
                break
            if current_iteration > maxiter:
                convergence = False
                break

        return {'success': convergence, 'x': betas, 'fun': res,
                'hess_inv': Hinv, 'nit': current_iteration}
