from xlogitprit import MultinomialLogit

import numpy as np
import pandas as pd
import time
import matplotlib.pyplot as plt

#To be provided by the user
df = pd.read_csv("https://raw.githubusercontent.com/timothyb0912/pylogit/master/examples/data/electricity_r_data_long.csv")
choice_id = df['chid']
ind_id =df['id']
varnames = ['cl','loc','wk','tod','seas']
# asvarnames = ['pf','cl','loc','wk','tod', 'seas']
# isvarnames = []
alternatives=[1,2,3,4]
choice_var = df['choice']
alt_var = df['alt']
R = 200
dist = ['n', 'ln', 'tn', 'u', 't', 'f']
#dist = ['n', 'ln', 'u', 'f']

model = MultinomialLogit()
# init_coeffs = np.repeat(.0, 6)
model.fit(X=df[varnames], y=choice_var, varnames=varnames, alts=alt_var, ids=choice_id, transformation="boxcox", transvars=['cl'],fit_intercept=False)
model.summary()