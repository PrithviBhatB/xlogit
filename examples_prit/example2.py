import pandas as pd
import numpy as np

from xlogitprit import MixedLogit
df = pd.read_csv("https://raw.githubusercontent.com/arteagac/xlogit/master/examples/data/electricity_long.csv")

varnames = ["pf", "cl", "loc", "wk", "tod", "seas"]

# df['tod'] = -df['tod']
# df['seas'] = -df['seas']


X = df[varnames].values
y = df['choice'].values

choice_id = df['chid']
alt = [1, 2, 3, 4]
np.random.seed(123)
model = MixedLogit()
model.fit(X, y,
          varnames,
          alts=alt,
          randvars={"cl": 'n', "loc": 'n', "wk": 'n', "tod": 'n', "seas": 'n'},
          # fit_intercept=True,
          # isvars=['pf'],
          # transformation="boxcox",
        #   transvars=['pf', 'cl', 'loc'],
          correlation=True,
          # weights=np.ones(361),
          # ids=choice_id,
          panels=df.id.values,
          isvars=[],
          # grad=False,
          # hess=False,
          # ftol=1e-5,
          # gtol=1e-3,
          halton=True,
          # method='L-BFGS-B',
          # maxiter=100,
          n_draws=400,
          # verbose=False
          )
model.summary()
model.corr()