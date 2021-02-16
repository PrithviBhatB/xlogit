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
np.random.seed(123)
model = MixedLogit()
model.fit(X, y,
          varnames,
          alts=alt,
          randvars={'cl': 'n', 'loc': 'n', 'wk': 'u', 'tod': 'ln', 'seas': 'ln'},
          # fit_intercept=True,
          # isvars=['pf'],
          # transformation="boxcox",
          # transvars=['cl', 'loc', 'wk'],
          correlation=True,
          # ids=choice_id,
          panels=df.id.values,
          isvars=[],
          # grad=False,
          # hess=False,
          tol=1e-8,
          halton=False,
          method='L-BFGS-B',
          n_draws=100,
          # verbose=False
          )
model.summary()