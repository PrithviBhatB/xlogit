import pandas as pd
import numpy as np

from xlogitprit import MultinomialLogit

df = pd.read_csv("https://raw.githubusercontent.com/arteagac/xlogit/master/examples/data/fishing_long.csv")


varnames = ['price', 'catch']
X = df[varnames]
y = df['choice']
asvarnames = ['price', 'catch']
isvarnames = []
rand_vars = {'price': 'n'}
alts = [1, 2, 3, 4]
# choice_id = df['chid']

model = MultinomialLogit()

model.fit(X, y, varnames=varnames, alts=alts,
          isvars=isvarnames, transvars=['price'], #randvars=rand_vars, #transvars=['price', 'catch'],
          fit_intercept=True, tol=1e-4#, hess=False, grad=False
          )
model.summary()