import pandas as pd
import numpy as np

from xlogitprit import MixedLogit
df = pd.read_csv("https://raw.githubusercontent.com/arteagac/xlogit/master/examples/data/fishing_long.csv")

varnames = ['price', 'catch']
X = df[varnames].values
y = df['choice'].values

model = MixedLogit()
initial_coeffs = [0.5, 0.5, 0.5, -0.1, 0.5, 1, 1, 1]
model.fit(X, y, varnames=varnames,
          # isvars=['income'],
          alts=['beach', 'boat', 'charter', 'pier'],
          randvars={'price': 'n'},
          transvars=['price', 'catch'],
          tol=1e-4,
        #   verbose=0,
          n_draws=600,
          maxiter=2000,
          # grad=False,
          # hess=False,
          halton=False,
        #   init_coeff=initial_coeffs,
          # correlation=True,
        #   method="Nelder-Mead",
        #   fit_intercept=True
          )
model.summary()