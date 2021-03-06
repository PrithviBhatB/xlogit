from xlogitprit import MixedLogit
import numpy as np
import pandas as pd
import time
import matplotlib.pyplot as plt

df = pd.read_csv("Final_RT_Dataset_weights_corrected.csv")

all_vars = ['cost', 'risk', 'seats', 'noise', 'crowdness', 'convloc', 'clientele']
is_vars = ['gender', 'age']
rand_vars = {'cost': 'n', 'risk': 'n', 'seats': 'n'}
choice_set = ['WaterTaxi', 'Ferry', 'Hovercraft', 'Helicopter']
choice_var = df['choice']
alt_var = df['idx.alt']
Tol = 1e-4
# av = df['av']
choice_id = df['idx.chid']
ind_id = df['idx.id']
weights = df['weight']

R = 100
model = MixedLogit()
model.fit(X=df[all_vars],
          y=choice_var,
          varnames=all_vars,
          alts=alt_var,
        #   isvars=is_vars,
          ids=choice_id,
          weights=weights,
        #   panels=ind_id,
          randvars=rand_vars,
        #   n_draws=R,
        #   correlation=['cost', 'risk'],
        #   fit_intercept=True,
        #   avail=av,
          gtol=Tol
          )
model.summary()