"""
Implements multinomial and mixed logit models
"""
# pylint: disable=invalid-name

from math import e
import numpy as np
from numpy.core.fromnumeric import var
from scipy.stats import t
from time import time
from abc import ABC, abstractmethod
from .boxcox_functions import boxcox_transformation, boxcox_param_deriv

class ChoiceModel(ABC):
    """Base class for estimation of discrete choice models"""

    def __init__(self):
        """Init Function

        Parameters
        ----------
        random_state: an integer used as seed to generate numpy random numbers
        """
        self.coeff_names = None
        self.coeff_ = None
        self.stderr = None
        self.zvalues = None
        self.pvalues = None
        self.loglikelihood = None
        self.initialData = None
        self.numFixedCoeffs = None
        self.numTransformedCoeffs = None

    def _reset_attributes(self):
        self.coeff_names = None
        self.coeff_ = None
        self.stderr = None
        self.zvalues = None
        self.pvalues = None
        self.loglikelihood = None
        self.initialData = None
        self.numFixedCoeffs = None
        self.numTransformedCoeffs = None

    @abstractmethod
    def fit(self, X, y, varnames=None, alt=None, isvars=None, transvars=None,
            transformation=None, id=None,
            weights=None, base_alt=None, fit_intercept=False, init_coeff=None,
            maxiter=2000, random_state=None, correlation=None):
        pass

    def _as_array(self, X, y, varnames, alt, isvars, transvars, id, weights, panel):
        X = np.asarray(X)
        y = np.asarray(y)
        initialData = X
        varnames = np.asarray(varnames) if varnames is not None else None
        alt = np.asarray(alt) if alt is not None else None
        isvars = np.asarray(isvars) if isvars is not None else None
        transvars = np.asarray(transvars) if transvars is not None else None
        id = np.asarray(id) if id is not None else None
        weights = np.asarray(weights) if weights is not None else None
        panel = np.asarray(panel) if panel is not None else None
        return X, y, initialData, varnames, alt, isvars, transvars, id, weights, panel

    def _pre_fit(self, alt, varnames, isvars, transvars, base_alt,
                 fit_intercept, transformation, maxiter):
        self._reset_attributes()
        self._fit_start_time = time()
        self.isvars = [] if isvars is None else isvars
        self.transvars = [] if transvars is None else transvars
        self.asvars = [v for v in varnames if (v not in self.isvars) and (v not in self.transvars)]
        self.alternatives = np.unique(alt)
        self.numFixedCoeffs = (len(self.isvars) + len(self.asvars)
        if not fit_intercept
        else (len(self.alternatives)-1)*(len(self.isvars)+1) + len(self.asvars))
        self.numTransformedCoeffs = len(self.transvars)*2 #trans var + lambda
        self.varnames = list(varnames)  # Easier to handle with lists
        self.fit_intercept = fit_intercept
        self.transformation = transformation
        self.base_alt = self.alternatives[0] if base_alt is None else base_alt
        self.correlation = False if correlation is None else correlation
        self.maxiter = maxiter

    def _post_fit(self, optimization_res, coeff_names, sample_size, verbose=1):
        self.convergence = optimization_res['success']
        self.coeff_ = optimization_res['x']
        self.stderr = np.sqrt(np.diag(optimization_res['hess_inv']))
        self.zvalues = np.nan_to_num(self.coeff_/self.stderr)
        self.pvalues = 2*t.pdf(-np.abs(self.zvalues), df=sample_size)
        self.loglikelihood = -optimization_res['fun']
        self.coeff_names = coeff_names
        self.total_iter = optimization_res['nit']
        self.estim_time_sec = time() - self._fit_start_time
        self.sample_size = sample_size
        self.aic = 2*len(self.coeff_) - 2*self.loglikelihood
        self.bic = np.log(sample_size)*len(self.coeff_) - 2*self.loglikelihood

        if not self.convergence and verbose > 0:
            print("**** The optimization did not converge after {} "
                  "iterations. ****".format(self.total_iter))
            print("Message: "+optimization_res['message'])


    def _setup_design_matrix(self, X):
        J = len(self.alternatives)
        N = int(len(X)/J)
        self.N = N
        self.J = J
        self.P = 1 # panel size, used in create_final_matrix
        self.R = 100
        isvars = self.isvars.copy()
        asvars = self.asvars.copy()
        transvars = self.transvars.copy()
        varnames = self.varnames.copy()
        self.varnames = np.array(varnames)
        if self.fit_intercept:
            isvars = np.insert(isvars, 0, '_inter')
            varnames.insert(0, '_inter')
            X = np.hstack((np.ones(J*N)[:, None], X))
        
        if self.transformation:
            self.transFunc = boxcox_transformation
            self.transform_deriv = boxcox_param_deriv

        ispos = [varnames.index(i) for i in isvars]  # Position of IS vars
        aspos = [varnames.index(i) for i in asvars]  # Position of AS vars
        transpos = [varnames.index(i) for i in transvars]  # Position of trans vars

        # Create design matrix
        # For individual specific variables
        if len(isvars):
            # Create a dummy individual specific variables for the alt
            dummy = np.tile(np.eye(J), reps=(N, 1))
            # Remove base alternative
            dummy = np.delete(dummy,
                              np.where(self.alternatives == self.base_alt)[0],
                              axis=1)
            Xis = X[:, ispos]
            # Multiply dummy representation by the individual specific data
            Xis = np.einsum('nj,nk->njk', Xis, dummy)
            self.Xis = Xis.reshape(N, J, (J-1)*len(ispos))
        else:
            self.Xis = np.array([])
        # For alternative specific variables
        self.Xas = None
        if asvars:
            Xas = X[:, aspos]
            self.Xas = Xas.reshape(N, J, -1)

        self.Xr = None
        if len(self.randvars):
            Xr = X[:, randpos]
            self.Xr = Xr.reshape(N, J, -1)

        self.Xf_trans = None
        self.Xr_trans = None
        self.Xtrans = None
        #  For variables to transform
        if len(transvars):
            if (self.Krtrans):
                    Xr_trans = X[:, randtranspos]
                    self.Xr_trans = Xr_trans.reshape(N, J, -1)

            if (self.Kftrans):
                Xf_trans = X[:, fixedtranspos]
                self.Xf_trans = Xf_trans.reshape(N, J, -1)

            if(not len(self.randtransvars) and not len(self.fixedtransvars)):
                Xtrans = X[:, transpos]
                self.Xtrans = Xtrans[:, len(transpos) - 1]

        #  For variables to transform
        if len(transvars):
            Xtrans = X[:, transpos]
            Xtrans = Xtrans[:, len(transpos) - 1]

        # Set design matrix based on existance of asvars and isvars
        if len(asvars) and len(isvars):
            X = np.dstack((Xis, Xas))
        elif len(asvars):
            X = Xas
        elif len(isvars):
            X = Xis

        if (self.Kr + self.Kftrans + self.Krtrans) > 0:
            if self.Kf !=0:
                self.Xf = create_final_matrix(self.Xf, self.Kf) #Data for fixed coeff
            if self.Kr !=0:
                self.Xr = create_final_matrix(self.Xr,self.Kr) #Data for random coeff
            if self.Kftrans != 0:
                self.Xf_trans = create_final_matrix(self.Xf_trans, self.Kftrans) #Data for fixed coeff
            if self.Krtrans != 0:
                self.Xr_trans = create_final_matrix(self.Xr_trans, self.Krtrans) #Data for random coeff
        intercept_names = ["_intercept.{}".format(j) for j in self.alternatives
                            if j != self.base_alt] if self.fit_intercept else []
        names = ["{}.{}".format(isvar, j) for isvar in isvars
                 for j in self.alternatives if j != self.base_alt]
        lambda_names = ["lambda.{}".format(transvar) for transvar in transvars]
        names = np.concatenate((names, transvars, lambda_names))
        names = np.array(names)

        return X, names

    def _check_long_format_consistency(self, id, alt, sorted_idx):
        alt = alt[sorted_idx]
        uq_alt = np.unique(alt)
        expect_alt = np.tile(uq_alt, int(len(id)/len(uq_alt)))
        if not np.array_equal(alt, expect_alt):
            raise ValueError('inconsistent alt values in long format')
        _, obs_by_id = np.unique(id, return_counts=True)
        if not np.all(obs_by_id/len(uq_alt)):  # Multiple of J
            raise ValueError('inconsistent alt and id values in long format')

    def _arrange_long_format(self, X, y, id, alt, panel=None):
        if id is not None:
            pnl = panel if panel is not None else np.ones(len(id))
            alt = alt.astype(str)
            alt = alt if len(alt) == len(id)\
                else np.tile(alt, int(len(id)/len(alt)))
            cols = np.zeros(len(id), dtype={'names': ['panel', 'id', 'alt'],
                                            'formats': ['<f4', '<f4', '<U64']})
            cols['panel'], cols['id'], cols['alt'] = pnl, id, alt
            sorted_idx = np.argsort(cols, order=['panel', 'id', 'alt'])
            X, y = X[sorted_idx], y[sorted_idx]
            if panel is not None:
                panel = panel[sorted_idx]
            self._check_long_format_consistency(id, alt, sorted_idx)
        return X, y, panel

    def _validate_inputs(self, X, y, alt, varnames, isvars, id, weights, panel,
                         base_alt, fit_intercept, max_iterations):
        if varnames is None:
            raise ValueError('The parameter varnames is required')
        if alt is None:
            raise ValueError('The parameter alternatives is required')
        if X.ndim != 2:
            raise ValueError("X must be an array of two dimensions in "
                             "long format")
        if y.ndim != 1:
            raise ValueError("y must be an array of one dimension in "
                             "long format")
        if len(varnames) != X.shape[1]:
            raise ValueError("The length of varnames must match the number "
                             "of columns in X")

    def summary(self):
        """
        Prints in console the coefficients and additional estimation outputs
        """
        print('fixed', self.numFixedCoeffs, 'trans', self.numTransformedCoeffs)
        print('coeff_names', self.coeff_names)
        if self.coeff_ is None:
            print("The current model has not been yet estimated")
            return
        if not self.convergence:
            print("-"*50)
            print("WARNING: Convergence was not reached during estimation. "
                  "The given estimates may not be reliable")
            print('*'*50)
        print("Estimation time= {:.1f} seconds".format(self.estim_time_sec))
        print("-"*75)
        print("{:19} {:>13} {:>13} {:>13} {:>13}"
              .format("Coefficient", "Estimate", "Std.Err.", "z-val", "P>|z|"))
        print("-"*75)
        fmt = "{:19} {:13.10f} {:13.10f} {:13.10f} {:13.3g} {:3}"
        for i in range(len(self.coeff_)):
            signif = ""
            if self.pvalues[i] < 0.001:
                signif = "***"
            elif self.pvalues[i] < 0.01:
                signif = "**"
            elif self.pvalues[i] < 0.05:
                signif = "*"
            elif self.pvalues[i] < 0.1:
                signif = "."
            print(fmt.format(self.coeff_names[i][:19], self.coeff_[i],
                             self.stderr[i], self.zvalues[i], self.pvalues[i],
                             signif))
        print("-"*75)
        print("Significance:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1")
        print("")
        print("Log-Likelihood= {:.3f}".format(self.loglikelihood))
        print("AIC= {:.3f}".format(self.aic))
        print("BIC= {:.3f}".format(self.bic))
