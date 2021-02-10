import pandas as pd
import numpy as np

# A simple multinomial logit model is estimated to check which variables require transformation when lognormal distribution is applied
from xlogitprit import MixedLogit
df = pd.read_csv("https://raw.githubusercontent.com/arteagac/xlogit/master/examples/data/electricity_long.csv")
varnames = ['cl', 'loc', 'pf', 'wk', 'tod', 'seas']
y = df['choice'].values
# choice_var = df['choice']
df['seas'] = -df['seas']
df['tod'] = -df['tod']
X = df[varnames].values

# isvarnames = []

randvars = {'cl': 'n', 'loc': 'ln', 'wk': 'n', 'tod': 'n', 'seas': 'n'}
alts = [1, 2, 3, 4]
np.random.seed(123)
model = MixedLogit()
# init_coeff=[]
model.fit(X, y,
          varnames=varnames,
          alts=alts,
        #   isvars=None,
          randvars=randvars,
        #   ids=df['chid'],
          panels=df.id.values,
          isvars=[],
          halton=True,
          correlation=True,
          transvars=['cl', 'loc'],
        #   method="L-BFGS-B",
        #   tol=1e-3,
        #   hess=False,
        #   grad=False,
        #   weights=None,
        #   avail=None,
        #   base_alt=None,
        #   fit_intercept=False,
        #   init_coeff=None,
        #   maxiter=2000,
        #   random_state=None,
        #   verbose=1
          n_draws=400
)
model.summary()