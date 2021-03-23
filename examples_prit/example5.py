import pandas as pd
import numpy as np

from xlogitprit import MixedLogit
from xlogitprit import MultinomialLogit
df = pd.read_csv("https://raw.githubusercontent.com/arteagac/xlogit/master/examples/data/electricity_long.csv")

varnames = ["pf", "cl", "loc", "wk", "tod", "seas"]

# df['tod'] = -df['tod']
# df['seas'] = -df['seas']
# print('sum', sum(np.where(df)))
# print("df['seas']", df['seas'])
# print(1/0)
X = df[varnames].values
y = df['choice'].values
choice_id = df['chid']
alt = [1, 2, 3, 4]
np.random.seed(123)

print('covariance', np.cov(np.transpose(X)))

# print(1/0)

model = MultinomialLogit()
model.fit(X, y,
          varnames,
          alts=alt,
        #   randvars={'seas': 'ln', 'wk': 'n', 'pf': 'n', 'loc': 'n'},
          fit_intercept=True,
        #   transformation="boxcox",
        #   transvars=['wk', 'seas'],
        #   correlation=True,
          # ids=choice_id,
        #   panels=df.id.values,
          # tol=1e-4,
          # grad=False,
          # hess=False,
          isvars=[],
        #   verbose=1,
          # halton=False,
        #   method='L-BFGS-B',
        #   n_draws=600
          )
model.summary()