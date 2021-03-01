from xlogitprit import MixedLogit

#import libraries

import numpy as np
import pandas as pd

df = pd.read_csv("Final_HBW_WC_Long.csv")

# Accessibility Time (One coefficient per alternative)
df['ACT_PT'] = df['act']*((df['alt'] == 'w2pt')|(df['alt'] == 'pr') | (df['alt'] == 'kr'))

# Waiting Time (One coefficient per alternative)
df['WT_PT'] = df['wt']*((df['alt'] == 'w2pt')|(df['alt'] == 'pr') | (df['alt'] == 'kr'))

df['EMP_DENS'] = df['emp_dens']*((df['alt'] == 'w2pt')|(df['alt'] == 'pr') | (df['alt'] == 'kr'))

df['ADUL_VEH'] = df['adul_veh']*((df['alt'] == 'cad')|(df['alt'] == 'pr'))

#To be provided by the user
choice_id = df['TRIPID']
ind_id =df['TRIPID']
varnames = ['tt', 'WT_PT', 'EMP_DENS']
# asvarnames = ['tt','tc','ACT_PT', 'WT_PT', 'EMP_DENS','ADUL_VEH']
isvarnames = []
X = df[varnames].values
y = df['Chosen_Mode'].values
choice_set=['cad','cap','w2pt','pr','kr','cycle','walk']
choice_var = df['Chosen_Mode']
alt_var = df['alt']
randvars={'EMP_DENS': 'n', 'WT_PT': 'u'}
R = 200
Tol = 1e-6

model = MixedLogit()
# init_coeff = [-2, -4, -3, -2, -1, -1, 0, 0, -0, -0.01, 0.0001, -1.15]
model.fit(X=df[varnames], y=choice_var, varnames=varnames, 
          isvars=isvarnames, alts=alt_var, ids=choice_id,
          randvars=randvars, tol=Tol, fit_intercept=True)#, init_coeff=init_coeff, tol=1e-2) #hess=False, grad=False)
model.summary()