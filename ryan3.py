import pandas as pd
import numpy as np

from xlogitprit import MixedLogit
df = pd.read_csv("https://raw.githubusercontent.com/arteagac/xlogit/master/examples/data/fishing_long.csv")

varnames = ['price', 'catch']
X = df[varnames].values
y = df['choice'].values

model = MixedLogit()
model.fit(X, y, varnames=varnames, 
          alt=['beach', 'boat', 'charter', 'pier'],
          randvars = {'price': 'n', 'catch': 'n'},
          transvars=['price', 'catch'],
          # method="L-BFGS-B"
          fit_intercept = True
          )
model.summary()