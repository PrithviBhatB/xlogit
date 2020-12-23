"""
Implements multinomial and mixed logit models
"""
# pylint: disable=invalid-name

from math import e
import numpy as np
from numpy.core.fromnumeric import var
from numpy.lib.arraysetops import isin
from scipy.stats import t
import scipy as sc
from time import time
from abc import ABC, abstractmethod
from .boxcox_functions import boxcox_transformation, boxcox_param_deriv
from itertools import product as cproduct

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
        transvars = np.asarray(transvars) if transvars is not None else []
        id = np.asarray(id) if id is not None else None
        weights = np.asarray(weights) if weights is not None else None
        panel = np.asarray(panel) if panel is not None else None
        return X, y, initialData, varnames, alt, isvars, transvars, id, weights, panel

    def _pre_fit(self, alt, varnames, isvars, transvars, base_alt,
                 fit_intercept, transformation, maxiter, panel, correlation=None, randvars=None):
        self._reset_attributes()
        self._fit_start_time = time()
        self.isvars = [] if isvars is None else isvars
        self.transvars = [] if transvars is None else transvars
        self.randvars = [] if randvars is None else randvars
        self.asvars = [v for v in varnames if ((v not in self.isvars) and (v not in self.transvars) and (v not in self.randvars))]
        self.randtransvars = [] if transvars is None else []
        self.fixedtransvars = [] if transvars is None else []
        self.alternatives = np.unique(alt)
        self.numFixedCoeffs = ((len(self.alternatives)-1)*(len(self.isvars)) + len(self.asvars) # TODO: CHECK
        if not fit_intercept
        else (len(self.alternatives)-1)*(len(self.isvars)+1) + len(self.asvars)) #+ len(self.asvars))
        self.numTransformedCoeffs = len(self.transvars)*2 #trans var + sd? + lambda
        self.varnames = list(varnames)  # Easier to handle with lists
        self.fit_intercept = fit_intercept
        self.transformation = transformation
        self.base_alt = self.alternatives[0] if base_alt is None else base_alt
        self.correlation = False if correlation is None else correlation
        self.maxiter = maxiter
        self.panel = panel

    def _post_fit(self, optimization_res, coeff_names, sample_size, verbose=1):
        self.convergence = optimization_res['success']
        self.coeff_ = optimization_res['x']
        if (isinstance(optimization_res['hess_inv'], sc.optimize.lbfgsb.LbfgsInvHessProduct)):
            hess = optimization_res['hess_inv'].todense()
        else:
            hess = optimization_res['hess_inv']
        self.stderr = np.sqrt(np.diag(np.array(hess)))
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
        N = P_N =  int(len(X)/J)
        self.P = 0
        self.N = N
        self.J = J
        if self.panel is not None:
            # Panel size.
            self.P_i = ((np.unique(self.panel, return_counts=True)[1])/J).astype(int)
            self.P = np.max(self.P_i)
            self.N = len(self.P_i)
        else:
            self.P = 1
            self.P_i = np.ones([N]).astype(int)
        isvars = self.isvars.copy()
        asvars = self.asvars.copy()
        transvars = self.transvars.copy()
        randvars = self.randvars.copy()
        randtransvars = self.randtransvars.copy()
        fixedtransvars = self.fixedtransvars.copy()
        varnames = self.varnames.copy()
        self.varnames = np.array(varnames)
        if self.fit_intercept:
            self.isvars = np.insert(np.array(self.isvars, dtype="<U16"), 0, '_inter')
            self.varnames = np.insert(np.array(self.varnames, dtype="<U16"), 0, '_inter')
            self.initialData = np.hstack((np.ones(J*N)[:, None], self.initialData))
            X = np.hstack((np.ones(J*N)[:, None], X))
        
        if self.transformation == "boxcox":
            self.transFunc = boxcox_transformation
            self.transform_deriv = boxcox_param_deriv

        # P_i = np.ones([self.N]).astype(int)
        S = np.zeros((self.N, self.P, self.J))
        for i in range(self.N):
            S[i, 0:self.P_i[i], :] = 1
        self.S = S
        ispos = [self.varnames.tolist().index(i) for i in self.isvars]  # Position of IS vars
        aspos = [self.varnames.tolist().index(i) for i in asvars]  # Position of AS vars
        randpos =  [self.varnames.tolist().index(i) for i in randvars]  # Position of AS vars
        transpos = [self.varnames.tolist().index(i) for i in transvars]  # Position of trans vars
        randtranspos = [self.varnames.tolist().index(i) for i in randtransvars] # bc transformed variables with random coeffs
        fixedtranspos = [self.varnames.tolist().index(i) for i in fixedtransvars] # bc transformed variables with fixed coeffs
        # if correlation = True correlation pos is randpos, if list get correct pos
        self.correlationpos = []
        if randvars:
            self.correlationpos = [self.randvars.index(i) for i in randvars] # Position of correlated variables within randvars
        if (isinstance(self.correlation, list)):
            self.correlationpos = [self.randvars.index(i) for i in self.correlation]
            self.uncorrelatedpos = [self.randvars.index(i) for i in self.randvars if i not in self.correlation]
        self.Kf = (J-1)*len(ispos) + len(aspos) #Number of fixed coefficients
        self.Kr = len(randpos)                     #Number of random coefficients
        self.Kftrans = len(fixedtranspos)   #Number of fixed coefficients of bc transformed vars
        self.Krtrans= len(randtranspos)   #Number of random coefficients of bc transformed vars
        self.Kchol = 0  # Number of random beta cholesky factors
        self.correlationLength = 0
        self.Kbw = self.Kr
        
        if (self.correlation):
            if (isinstance(self.correlation, list)):
                self.correlationLength = len(self.correlation)
                self.Kbw = self.Kr - len(self.correlation)
            else:
                self.correlationLength = self.Kr
                self.Kbw = 0
        if (self.correlation):
            if (isinstance(self.correlation, list)):
                self.Kchol = (len(self.correlation) * (len(self.correlation)+1))/2
            else:
                self.Kchol =  (len(self.randvars) * (len(self.randvars)+1))/2
        # Create design matrix
        # For individual specific variables
        self.Xis = None
        if len(self.isvars):
            # Create a dummy individual specific variables for the alt
            dummy = np.tile(np.eye(J), reps=(P_N, 1))
            # Remove base alternative
            dummy = np.delete(dummy,
                              np.where(self.alternatives == self.base_alt)[0],
                              axis=1)
            Xis = X[:, ispos]
            # Multiply dummy representation by the individual specific data
            Xis = np.einsum('nj,nk->njk', Xis, dummy)
            self.Xis = Xis.reshape(P_N, self.J, (self.J-1)*len(ispos))
        else:
            self.Xis = np.array([])
        # For alternative specific variables
        self.Xas = None
        if asvars:
            Xas = X[:, aspos]
            self.Xas = Xas.reshape(P_N, self.J, -1)

        self.Xr = None
        if len(self.randvars):
            Xr = X[:, randpos]
            self.Xr = Xr.reshape(P_N, self.J, -1)

        self.Xf_trans = None
        self.Xr_trans = None
        self.Xtrans = None
        #  For variables to transform
        if len(transvars):
            if (self.Krtrans):
                    Xr_trans = X[:, randtranspos]
                    self.Xr_trans = Xr_trans.reshape(P_N, J, -1)

            if (self.Kftrans):
                Xf_trans = X[:, fixedtranspos]
                self.Xf_trans = Xf_trans.reshape(P_N, J, -1)

            if(not len(self.randtransvars) and not len(self.fixedtransvars)):
                Xtrans = X[:, transpos]
                self.Xtrans = Xtrans[:, len(transpos) - 1]

        # Set design matrix based on existance of asvars and isvars
        self.Xf = []
        if len(self.asvars) and len(self.isvars):
            self.Xf = np.dstack((self.Xis, self.Xas))
        elif len(self.asvars):
            self.Xf = self.Xas
        elif len(self.isvars):
            self.Xf = self.Xis


        def create_final_matrix(design_matrix, num_col, isZero=True):
            X_Final = np.zeros((self.N, self.P, self.J, num_col)) if isZero \
                      else np.ones((self.N, self.P, self.J, num_col))
            k = 0
            while k < P_N:
                for i in range(self.N):
                    for j in range(self.P_i[i]):
                        X_Final[i,j,:,:] = design_matrix[k,:,:]
                        k = k+1
            return(X_Final)

        if (self.Kf + self.Kr + self.Kftrans + self.Krtrans) > 0:
            # TODO: BALANCE PANELS
            if self.Kf !=0:
                self.Xf = create_final_matrix(self.Xf, self.Kf) #Data for fixed coeff
            if self.Kr !=0:
                self.Xr = create_final_matrix(self.Xr,self.Kr) #Data for random coeff
            if self.Kftrans != 0:
                self.Xf_trans = create_final_matrix(self.Xf_trans, self.Kftrans) #Data for fixed coeff
            if self.Krtrans != 0:
                self.Xr_trans = create_final_matrix(self.Xr_trans, self.Krtrans) #Data for random coeff
        self.y = create_final_matrix((self.y.reshape(P_N, self.J, 1)), 1)
        
        
        def _balance_panels(self, X, y, panel):
            _, p_obs = np.unique(self.panel, return_counts=True)
            p_obs = (p_obs/J).astype(int)
            if X is None:
                return None
            if not np.all(p_obs[0] == p_obs):  # Balancing needed
                Xbal = np.zeros_like(X)
                self.panel_info = np.zeros((self.N, self.P))
                cum_p = 1 # Cumulative sum of n_obs at each iteration
                # TODO: Why = 1 works? (but original = 0 doesn't)
                for n, p in enumerate(p_obs):
                    Xbal[n*self.P:n*self.P +p, :, :, :] = X[cum_p:cum_p + p, :, :, :]
                    self.panel_info[n, :p] = np.ones(p)
                    cum_p += p
            else: # No balancing needed
                Xbal = X
                self.panel_info = np.ones((self.N, self.P))
            return Xbal

        if (self.Kf > 0):
            self.Xf = _balance_panels(self, self.Xf, self.y, self.panel)
        self.Xr = _balance_panels(self, self.Xr, self.y, self.panel)
        self.Xf_trans = _balance_panels(self, self.Xf_trans, self.y, self.panel)
        self.Xr_trans = _balance_panels(self, self.Xr_trans, self.y, self.panel)
        self.y = _balance_panels(self, self.y, self.y, self.panel)


        intercept_names = ["_intercept.{}".format(j) for j in self.alternatives
                            if j != self.base_alt] if self.fit_intercept else []
        names = ["{}.{}".format(isvar, j) for isvar in isvars
                 for j in self.alternatives if j != self.base_alt]
        lambda_names_fixed = ["lambda.{}".format(transvar) for transvar in fixedtransvars]
        lambda_names_rand = ["lambda.{}".format(transvar) for transvar in randtransvars]
        randvars = [x for x in self.randvars]
        chol =  ["chol." + self.randvars[self.correlationpos[i]] + "." + \
                    self.randvars[self.correlationpos[j]] for i \
                    in range(self.correlationLength) for j in range(i+1) ]
        br_w_names = ["sd." + x for x in self.randvars]
        if (isinstance(self.correlation, list)): #if not all r.v.s correlated...
            sd_uncorrelated_pos = [self.varnames.tolist().index(x) for x in self.varnames 
                        if x not in self.correlation and x in self.randvars]
            br_w_names = np.char.add("sd.", self.varnames[sd_uncorrelated_pos])
        sd_rand_trans = np.char.add("sd.", self.varnames[randtranspos])
        names = np.concatenate((intercept_names, names, asvars, randvars, chol, br_w_names,
        fixedtransvars, lambda_names_fixed, randtransvars, sd_rand_trans, lambda_names_rand))
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
                             signif
                             ))
        print("-"*75)
        print("Significance:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1")
        print("")
        print("Log-Likelihood= {:.3f}".format(self.loglikelihood))
        print("AIC= {:.3f}".format(self.aic))
        print("BIC= {:.3f}".format(self.bic))
