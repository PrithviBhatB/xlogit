import pandas as pd
import numpy as np

from xlogitprit import MixedLogit
df = pd.read_csv("https://raw.githubusercontent.com/arteagac/xlogit/master/examples/data/fishing_long.csv")

varnames = ['price', 'catch']
X = df[varnames].values
y = df['choice'].values

model = MixedLogit()
# initial_coeffs = [1, 2, 9, -0.25, 0.4, 0.1, 0.1, 0.5, 0.7]
model.fit(X, y, varnames=varnames,
          # isvars=['income'],
          alts=['beach', 'boat', 'charter', 'pier'],
          randvars={'price': 'n', 'catch': 'n'},
          transvars=['price', 'catch'],
          # init_coeff=initial_coeffs,
          # correlation=True,
          # method="L-BFGS-B",
          fit_intercept=True
          )
model.summary()