from xlogitprit.mixed_logit import MixedLogit
import numpy as np
import pandas as pd
import site

# print(site.getsitepackages()[0])
# print(1/0)
df = pd.read_csv("examples_prit/artificial.csv")

model = MixedLogit()
varnames = ['x1', 'x2', 'x3', 'x4', 'x5']
shorten_num = 3000
X = df[varnames].values[:shorten_num]
y = df['choice'].values[:shorten_num]

model.fit(X, y, ids=df['id'][:shorten_num], varnames=varnames,
          alts=df['alt'][:shorten_num], randvars={'x5': 'n'},
          transvars=['x4', 'x5'],
          method="L-BFGS-B"
          )
model.summary()
