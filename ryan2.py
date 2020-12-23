from xlogit import MixedLogit

import pandas as pd
import numpy as np
df = pd.read_csv("https://raw.githubusercontent.com/arteagac/xlogit/master/examples/data/electricity_long.csv")

varnames = ["pf", "cl", "loc", "wk", "tod", "seas"]
X = df[varnames].values
y = df['choice'].values
alt =[1, 2, 3, 4]

model = MixedLogit()
model.fit(X, y, 
          varnames, 
          alt=alt, 
          randvars={'pf': 'n','cl':'n','loc':'n','wk':'n','tod':'n','seas':'n'}, 
          transformation="boxcox",
          transvars=['cl', 'loc'],
          correlation=['wk', 'tod'],
          mixby=df.id.values,
          n_draws = 600)
model.summary()