import pandas as pd
import numpy as np
from xlogitprit import MixedLogit

df = pd.read_csv("https://raw.githubusercontent.com/arteagac/xlogit/master/examples/data/car100_long.csv")
df.price = -1*df.price/10000
df.operating_cost = -1*df.operating_cost

varnames = ['high_performance','medium_performance','price', 'operating_cost',
            'range', 'electric', 'hybrid'] 

X = df[varnames].values
y = df['choice'].values

model = MixedLogit()
model.fit(X, y, varnames = varnames,
          alts=['car','bus','bike'],
          randvars = {'price': 'ln', 'operating_cost': 'n',
                      'range': 'ln', 'electric':'n', 'hybrid': 'n'}, 
          panels=df.person_id.values, #Panel column
          n_draws = 200) 
model.summary()