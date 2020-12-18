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

    def fit(self, X, y, varnames=None, alt=None, isvars=None, transvars=None, transformation=None,
            id=None, weights=None, base_alt=None, fit_intercept=False,
            init_coeff=None, maxiter=2000, random_state=None, verbose=1):


<<<<<<< Updated upstream
        print('testryan1', 'fit', fit_intercept, 'transformation', transformation)
=======
>>>>>>> Stashed changes
        X, y, initialData, varnames, alt, isvars, transvars, id, weights, _\
            = self._as_array(X, y, varnames, alt, isvars, transvars, id, weights, None)
        self._validate_inputs(X, y, alt, varnames, isvars, id, weights, None,
                              base_alt, fit_intercept, maxiter)

<<<<<<< Updated upstream
        self._pre_fit(alt, varnames, isvars, transvars, base_alt, fit_intercept, transformation, maxiter)
        X, y, panel = self._arrange_long_format(X, y, id, alt)
        print('testryan2', 'fit', fit_intercept, 'transformation', transformation)
        self.initialData = initialData
        # self.fixedtransvars = self.transvars
        if random_state is not None:
            np.random.seed(random_state)
        print('hereinit', self.transformation)
        if transformation == "boxcox":
            print('here3')
            self.transFunc = boxcox_transformation
            self.transform_deriv = boxcox_param_deriv

=======
        self._pre_fit(alt, varnames, isvars, transvars, base_alt, transformation, fit_intercept, maxiter)
        X, y, panel = self._arrange_long_format(X, y, id, alt)

        self.initialData = initialData
        if random_state is not None:
            np.random.seed(random_state)
>>>>>>> Stashed changes
        if init_coeff is None:
            betas = np.repeat(.0, self.numFixedCoeffs + self.numTransformedCoeffs)
        else:
            betas = init_coeff
            if len(init_coeff) != X.shape[1]:
                raise ValueError("The size of initial_coeff must be: "
                                 + int(X.shape[1]))

        X, Xnames = self._setup_design_matrix(X)
        # add transformation vars and corresponding lambdas
        lambda_names = ["lambda.{}".format(transvar) for transvar in transvars]
        transnames = np.concatenate((transvars, lambda_names))
        Xnames = np.concatenate((Xnames, transnames))
        y = y.reshape(self.N, self.J)

        # Call optimization routine
        optimizat_res = self._bfgs_optimization(betas, X, y, weights, maxiter)
        self._post_fit(optimizat_res, Xnames, int(1182/4), verbose)

    def _compute_probabilities(self, betas, X):
<<<<<<< Updated upstream
        transpos = [self.varnames.tolist().index(i) for i in self.transvars]  # Position of trans vars
        X_trans = self.initialData[:, transpos]
        X_trans = X_trans.reshape(self.N, self.J, len(transpos))
        XB = 0
        if self.numFixedCoeffs > 0:
            B = betas[0:self.numFixedCoeffs]
            XB = self.Xf.dot(B)
=======
        transpos = [self.varnames.index(i) for i in self.transvars]  # Position of trans vars
        X_trans = self.initialData[:, transpos]
        X_trans = X_trans.reshape(X.shape[0], len(self.alternatives), len(transpos))
        XB = 0
        if self.numFixedCoeffs > 0:
            B = betas[0:self.numFixedCoeffs]
            XB = X.dot(B)
>>>>>>> Stashed changes
        Xtrans_lambda = None
        if self.numTransformedCoeffs > 0:
            B_transvars = betas[self.numFixedCoeffs:int(self.numFixedCoeffs+(self.numTransformedCoeffs/2))]
            lambdas = betas[int(self.numFixedCoeffs+(self.numTransformedCoeffs/2)):]
            # applying transformations
            Xtrans_lambda = self.transFunc(X_trans, lambdas)
            XB_trans = Xtrans_lambda.dot(B_transvars)
            XB += XB_trans
        XB[XB > 700] = 700 # avoiding infs
        XB[np.isposinf(XB)] = 1e+30 # avoiding infs
        XB[np.isneginf(XB)] = 1e-30 # avoiding infs
        eXB = np.exp(XB)
        p = eXB/np.sum(eXB, axis=1, keepdims=True)  # (N,J)
        return p, Xtrans_lambda

    def _loglik_and_gradient(self, betas, X, y, weights):
        p, Xtrans_lmda = self._compute_probabilities(betas, X)
<<<<<<< Updated upstream
=======
        print('p', p.shape)
>>>>>>> Stashed changes
        # Log likelihood
        lik = np.sum(y*p, axis=1)
        loglik = np.log(lik)
        if weights is not None:
            loglik = loglik*weights
        loglik = np.sum(loglik)
        print('loglik', loglik)
        # Individual contribution to the gradient

<<<<<<< Updated upstream
        transpos = [self.varnames.tolist().index(i) for i in self.transvars]  # Position of trans vars
        B_trans = betas[self.numFixedCoeffs:int(self.numFixedCoeffs+(self.numTransformedCoeffs/2))]
        lambdas = betas[int(self.numFixedCoeffs+(self.numTransformedCoeffs/2)):]
        X_trans = self.initialData[:, transpos]
        X_trans = X_trans.reshape(self.N, len(self.alternatives), len(transpos))
        if self.numFixedCoeffs > 0:
            grad = np.einsum('nj,njk -> nk', (y-p), self.Xf)
        else:
            grad = np.array([])
=======
        transpos = [self.varnames.index(i) for i in self.transvars]  # Position of trans vars
        B_trans = betas[self.numFixedCoeffs:int(self.numFixedCoeffs+(self.numTransformedCoeffs/2))]
        lambdas = betas[int(self.numFixedCoeffs+(self.numTransformedCoeffs/2)):]
        # print('X shape', X.shape)
        X_trans = self.initialData[:, transpos]
        X_trans = X_trans.reshape(X.shape[0], len(self.alternatives), len(transpos))
        # K = (len(self.alternatives)-1)*(len(self.isvars)+1) + len(self.asvars)  # Number of fixed coeffs
        if self.numFixedCoeffs > 0:
            grad = np.einsum('nj,njk -> nk', (y-p), X)
        else:
            grad = np.array([])
        # print('initgrad', grad)
>>>>>>> Stashed changes
        if self.numTransformedCoeffs > 0:
            # Individual contribution of trans to the gradient
            gtrans = np.einsum('nj, njk -> nk', (y - p), Xtrans_lmda)
            der_Xtrans_lmda = self.transform_deriv(X_trans, lambdas)
            der_XBtrans = np.einsum('njk,k -> njk', der_Xtrans_lmda, B_trans)
            gtrans_lmda = np.einsum('nj,njk -> nk', (y - p), der_XBtrans)
            grad = np.concatenate((grad, gtrans, gtrans_lmda), axis=1) if grad.size else np.concatenate((gtrans, gtrans_lmda), axis=1) # (N, K)
<<<<<<< Updated upstream
=======

>>>>>>> Stashed changes
        if weights is not None:
            grad = grad*weights[:, None]

        H = np.dot(grad.T, grad)
        Hinv = np.linalg.pinv(H)
        grad = np.sum(grad, axis=0)
<<<<<<< Updated upstream
=======
        print('Hinv', Hinv)
        print('grad', grad)
>>>>>>> Stashed changes
        return (-loglik, -grad, Hinv)

    def _ryan_optimization(self, betas, X, y, weights, maxiter):
        res_init, g, Hinv = self._loglik_and_gradient(betas, X, y, weights)
        res = minimize(self._loglik_and_gradient, betas, args=(X, y, weights), jac=True, method='BFGS', tol=1e-10, options={'gtol': 1e-10})
        return res

    def _bfgs_optimization(self, betas, X, y, weights, maxiter):
        print('bfgsbetas', betas)
        res, g, Hinv = self._loglik_and_gradient(betas, X, y, weights)
        current_iteration = 0
        convergence = False
        # betas = np.zeros(10)
        while True:
            old_g = g
<<<<<<< Updated upstream
            d = -Hinv.dot(g)
=======
            # print('g', g)
            d = -Hinv.dot(g)
            # print('d', d)
>>>>>>> Stashed changes
            step = 2
            while True:
                step = step/2
                s = step*d
                # s[s == 0] = 1e+30
                betas = betas + s
                # print('s', s)
                # print('betas step', betas)
                resnew, gnew, _ = self._loglik_and_gradient(betas, X, y,
                                                            weights)
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

<<<<<<< Updated upstream
            # print('Hinvhere', Hinv)
=======
            print('Hinvhere', Hinv)
>>>>>>> Stashed changes
            current_iteration = current_iteration + 1
            if np.abs(res - old_res) < 0.00001:
                convergence = True
                break
            if current_iteration > maxiter:
                convergence = False
                break

        return {'success': convergence, 'x': betas, 'fun': res,
                'hess_inv': Hinv, 'nit': current_iteration}
