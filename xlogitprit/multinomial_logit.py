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


        print('testryan1', 'fit', fit_intercept, 'transformation', transformation)
        X, y, initialData, varnames, alt, isvars, transvars, id, weights, panel, \
            = self._as_array(X, y, varnames, alt, isvars, transvars, id, weights, None)
        self._validate_inputs(X, y, alt, varnames, isvars, id, weights, None,
                              base_alt, fit_intercept, maxiter)

        self._pre_fit(alt, varnames, isvars, transvars, base_alt, fit_intercept, transformation, maxiter, panel)
        X, y, panel = self._arrange_long_format(X, y, id, alt)
        self.y = y
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
        transpos = [self.varnames.tolist().index(i) for i in self.transvars]  # Position of trans vars
        X_trans = self.initialData[:, transpos]
        X_trans = X_trans.reshape(self.N, self.J, len(transpos))
        XB = 0
        if self.numFixedCoeffs > 0:
            B = betas[0:self.numFixedCoeffs]
            print('self.Kf', self.Kf)
            print('self.Xf', self.Xf.shape, 'B', B)
            XB = self.Xf.dot(B)
        Xtrans_lambda = None
        if self.numTransformedCoeffs > 0:
            B_transvars = betas[self.numFixedCoeffs:int(self.numFixedCoeffs+(self.numTransformedCoeffs/2))]
            lambdas = betas[int(self.numFixedCoeffs+(self.numTransformedCoeffs/2)):]
            # applying transformations
            Xtrans_lambda = self.transFunc(X_trans, lambdas)
            XB_trans = Xtrans_lambda.dot(B_transvars)
            print('XB', XB.shape, 'XB_trans', XB_trans.shape)
            XB_trans = XB_trans.reshape(self.N, 1, self.J)
            print('XB', XB.shape, 'XB_trans', XB_trans.shape)
            XB += XB_trans
        XB[XB > 700] = 700 # avoiding infs
        XB[np.isposinf(XB)] = 1e+30 # avoiding infs
        XB[np.isneginf(XB)] = 1e-30 # avoiding infs
        XB = XB.reshape(self.N, self.J)
        print('XB', XB.shape)
        eXB = np.exp(XB)
        print('eXB', eXB)
        p = eXB/np.sum(eXB, axis=1, keepdims=True)  # (N,J)
        # p[np.isnan(p)] = 1e-10
        print('ppp', p)
        return p, Xtrans_lambda

    def _loglik_and_gradient(self, betas, X, y, weights):
        p, Xtrans_lmda = self._compute_probabilities(betas, X)
        # Log likelihood
        lik = np.sum(y*p, axis=1)
        loglik = np.log(lik)
        if weights is not None:
            loglik = loglik*weights
        loglik = np.sum(loglik)
        # Individual contribution to the gradient

        transpos = [self.varnames.tolist().index(i) for i in self.transvars]  # Position of trans vars
        B_trans = betas[self.numFixedCoeffs:int(self.numFixedCoeffs+(self.numTransformedCoeffs/2))]
        lambdas = betas[int(self.numFixedCoeffs+(self.numTransformedCoeffs/2)):]
        X_trans = self.initialData[:, transpos]
        X_trans = X_trans.reshape(self.N, len(self.alternatives), len(transpos))
        p = p.reshape(self.N, self.J)
        # print('y', y.shape, 'p', p.shape)
        ymp = y - p
        if self.numFixedCoeffs > 0:
            print('self.Xf', self.Xf.shape)
            grad = np.einsum('nj,nijk -> nk', ymp, self.Xf)
            # print('grad1', grad.shape)
        else:
            grad = np.array([])
        if self.numTransformedCoeffs > 0:
            # Individual contribution of trans to the gradient
            print('Xtrans_lmda', Xtrans_lmda.shape)
            gtrans = np.einsum('nj,njk -> nk',ymp, Xtrans_lmda)
            der_Xtrans_lmda = self.transform_deriv(X_trans, lambdas)
            der_XBtrans = np.einsum('njk,k -> njk', der_Xtrans_lmda, B_trans)
            gtrans_lmda = np.einsum('nj,njk -> nk', ymp, der_XBtrans)
            print('grad', grad.shape, 'gtrans', gtrans.shape, 'gtrans_lmda', gtrans_lmda.shape)
            grad = np.concatenate((grad, gtrans, gtrans_lmda), axis=1) if grad.size else np.concatenate((gtrans, gtrans_lmda), axis=1) # (N, K)
            print('grad2', grad.shape)
        if weights is not None:
            grad = grad*weights[:, None]

        H = np.dot(grad.T, grad)
        print('H', H)
        Hinv = np.linalg.pinv(H)
        grad = np.sum(grad, axis=0)
        print('grad', grad)
        return (-loglik, -grad, Hinv)

    def _ryan_optimization(self, betas, X, y, weights, maxiter):
        res_init, g, Hinv = self._loglik_and_gradient(betas, X, y, weights)
        res = minimize(self._loglik_and_gradient, betas, args=(X, y, weights), jac=True, method='BFGS', tol=1e-10, options={'gtol': 1e-10})
        return res

    def _bfgs_optimization(self, betas, X, y, weights, maxiter):
        res, g, Hinv = self._loglik_and_gradient(betas, X, y, weights)
        current_iteration = 0
        convergence = False
        # betas = np.zeros(10)
        while True:
            old_g = g
            d = -Hinv.dot(g)
            step = 2
            while True:
                step = step/2
                s = step*d
                betas = betas + s
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
            current_iteration = current_iteration + 1
            if np.abs(res - old_res) < 0.00001:
                convergence = True
                break
            if current_iteration > maxiter:
                convergence = False
                break

        return {'success': convergence, 'x': betas, 'fun': res,
                'hess_inv': Hinv, 'nit': current_iteration}
