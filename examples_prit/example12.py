from xlogitprit import MixedLogit
import numpy as np
import pandas as pd
import time

df = pd.read_csv("examples_prit/Final_RT_Dataset_weights_corrected.csv")

all_vars = ['cost', 'risk', 'seats', 'noise', 'crowdness', 'convloc', 'clientele']
is_vars = ['gender', 'age']
rand_vars = {'cost': 'n', 'risk': 'n', 'seats': 'ln'}
choice_set = ['WaterTaxi', 'Ferry', 'Hovercraft', 'Helicopter']
choice_var = df['choice']
alt_var = df['idx.alt']
Tol = 1e-4
# av = df['av']
choice_id = df['idx.chid']
ind_id = df['idx.id']
weights = df['weight']

R = 100
X = df[all_vars]
# X_zeros = np.zeros_like(X)
model = MixedLogit()
model.fit(X=X,
          y=choice_var,
          varnames=all_vars,
          alts=alt_var,
        #   isvars=is_vars,
          ids=choice_id,
          weights=weights,
          # init_coeff=np.repeat(20, 10),
        #   panels=ind_id,
          randvars=rand_vars,
        #   n_draws=R,
        #   correlation=['cost', 'risk'],
        #   fit_intercept=True,
        #   avail=av,
          gtol=Tol
          )
model.summary()