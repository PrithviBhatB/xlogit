from xlogitprit import MixedLogit
import numpy as np
import pandas as pd
data_file = "https://raw.githubusercontent.com/arteagac/xlogit/master/examples/data/fishing_long.csv"
df = pd.read_csv(data_file)

varnames = ['price', 'catch', 'income']
X = df[varnames].values
y = df['choice'].values
transvars = ['catch']

model = MixedLogit()
model.fit(
  X,
  y,
  # isvars=['income'],
  transvars=transvars,
  transformation="boxcox",
  alts=['beach', 'boat', 'charter', 'pier'],
  # scipy_optimisation=True,
  isvars=['income', 'price'],
  randvars={'catch': 'n'},
  weights=np.ones(1182),
  # init_coeff=np.repeat(0, 11),
  # gtol=1e-4,
  # grad=False,
  # hess=False,
  # tol=1e-4,
  # method="L-BFGS-B",
  fit_intercept=True,
  varnames=varnames
)
model.summary()
