import pandas as pd
import numpy as np

from xlogitprit import MixedLogit

df = pd.read_csv("https://raw.githubusercontent.com/arteagac/xlogit/master/examples/data/fishing_long.csv")


varnames = ['price', 'catch', 'income']
X = df[varnames]
y = df['choice']
asvarnames = ['price', 'catch']
isvarnames = ['income']
rand_vars = {'price': 'n'}
alts = [1, 2, 3, 4]
# choice_id = df['chid']

model = MixedLogit()

model.fit(X, y, varnames=varnames, alts=alts,
          isvars=isvarnames, randvars=rand_vars,
          fit_intercept=True, tol=1e-2
          )
model.summary()