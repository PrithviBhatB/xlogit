import pandas as pd
import numpy as np

from xlogitprit import MixedLogit
df = pd.read_csv("https://raw.githubusercontent.com/arteagac/xlogit/master/examples/data/electricity_long.csv")

varnames = ["pf", "cl", "loc", "wk", "tod", "seas"]

df['tod'] = -df['tod']
df['seas'] = -df['seas']


X = df[varnames].values
y = df['choice'].values

choice_id = df['chid']
alt = [1, 2, 3, 4]

randvars1 = {"cl": 'n', "loc": 't', "wk": 'u', "tod": 'ln', "seas": 'ln'}
randvars2 = {"wk": 'u', "tod": 'ln', "seas": 'ln', "cl": 'n', "loc": 'n'}
randvars3 = {"wk": 'u', "tod": 'ln', "seas": 'ln', "cl": 'n', "loc": 'n'}

np.random.seed(123)
model = MixedLogit()
model.fit(X, y,
          varnames,
          alts=alt,
          randvars=randvars3,
          # fit_intercept=True,
          # isvars=['pf'],
          # transformation="boxcox",
          # transvars=['pf'],
          correlation=True,
        #   correlation=['wk', 'tod', 'loc'],
          # weights=np.ones(361),
          # ids=choice_id,
          panels=df.id.values,
          isvars=[],
          # grad=False,
          # hess=False,
          # ftol=1e-6,
          # gtol=1e-6,
          halton=True,
          # method='L-BFGS-B',
          # maxiter=100,
          n_draws=100,
          # verbose=False
          )
model.summary()
# model.corr()
# model.cov()
# model.stddev()
# indpar = model.fitted()
# print('indpar', indpar)