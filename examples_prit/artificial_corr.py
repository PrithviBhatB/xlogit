from xlogitprit.mixed_logit import MixedLogit
# import numpy as np
import pandas as pd
# import site

# print(site.getsitepackages()[0])
# print(1/0)
df = pd.read_csv("examples_prit/artificial_corr.csv")

model = MixedLogit()
varnames = ['price', 'time', 'conven', 'comfort', 'meals', 'petfr', 'emipp']
# shorten_num = 3000
X = df[varnames].values
y = df['choice'].values

model.fit(X, y, ids=df['id'], varnames=varnames,
          alts=df['alt'], randvars={'meals': 'n', 'petfr': 'n', 'emipp': 'n'},
        #   transvars=['x4', 'x5'],
          # method="L-BFGS-B",
          grad=False,
          hess=False,
        correlation=True
        )
model.summary()
model.corr()
model.cov()
model.stddev()
