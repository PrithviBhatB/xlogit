import pandas as pd
import numpy as np

from xlogitprit import MixedLogit
df = pd.read_csv("https://raw.githubusercontent.com/arteagac/xlogit/master/examples/data/electricity_long.csv")

varnames = ["pf", "cl", "loc", "wk", "tod", "seas"]

df['tod'] = -df['tod']
df['seas'] = -df['seas']


X = df[varnames].values
y = df['choice'].values
alt = [1, 2, 3, 4]
np.random.seed(123)
model = MixedLogit()
model.fit(X, y,
          varnames,
          alts=alt,
          randvars={'cl': 'n', 'loc': 'n', 'wk': 'u', 'tod': 'ln','seas': 'ln'},
          # fit_intercept=True,
          # transformation="boxcox",
          # transvars=['cl', 'loc', 'wk'],
          correlation=['cl', 'loc', 'wk'],
          panels=df.id.values,
          # halton=False,
          # method='L-BFGS-B',
          n_draws=600)
model.summary()