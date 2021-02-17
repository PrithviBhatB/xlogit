from xlogitprit import MultinomialLogit
import numpy as np
import pandas as pd
data_file = "https://raw.githubusercontent.com/arteagac/xlogit/master/examples/data/fishing_long.csv"
df = pd.read_csv(data_file)

varnames = ['price', 'catch']
X = df[varnames].values
y = df['choice'].values
transvars = ['price']

model = MultinomialLogit()
model.fit(
  X,
  y,
  # isvars=['income'],
  # transvars=['price'],
  # transformation="boxcox",
  alts=['beach', 'boat', 'charter', 'pier'],
  # init_coeff=np.repeat(.1, 8),
  # scipy_optimisation=True,
  # isvars=['income'],
  # grad=False,
  # hess=False,
  tol=1e-3,
  # method="L-BFGS-B",
  fit_intercept=True,
  varnames=varnames
)
model.summary()
