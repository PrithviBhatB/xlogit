import pandas as pd
import numpy as np

from xlogitprit import MultinomialLogit

df = pd.read_csv("https://raw.githubusercontent.com/arteagac/xlogit/master/examples/data/electricity_long.csv")

varnames = ["pf", "cl", "loc", "wk", "seas"]

X = df[varnames].values
y = df['choice'].values
# df['seas'] = -df
choice_id = df['chid']
alts = [1, 2, 3, 4]
np.random.seed(123)
model = MultinomialLogit()

maxiter = 1000

model.fit(X=X, y=y, varnames=varnames, isvars=[], alts=alts, fit_intercept=True,
          transformation="boxcox", maxiter=maxiter, tol=1e-3, method="L-BFGS-B")

model.summary()