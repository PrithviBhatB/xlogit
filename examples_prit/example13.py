from xlogitprit import MultinomialLogit, MixedLogit

#import libraries

import numpy as np
import pandas as pd

df = pd.read_csv("examples_prit/Final_HBW_WC_Long.csv")

# Accessibility Time (One coefficient per alternative)
df['ACT_PT'] = df['act']*((df['alt'] == 'w2pt')|(df['alt'] == 'pr') | (df['alt'] == 'kr'))

# Waiting Time (One coefficient per alternative)
df['WT_PT'] = df['wt']*((df['alt'] == 'w2pt')|(df['alt'] == 'pr') | (df['alt'] == 'kr'))

df['EMP_DENS'] = df['emp_dens']*((df['alt'] == 'w2pt')|(df['alt'] == 'pr') | (df['alt'] == 'kr'))

df['ADUL_VEH'] = df['adul_veh']*((df['alt'] == 'cad')|(df['alt'] == 'pr'))

#To be provided by the user
choice_id = df['TRIPID']
ind_id =df['TRIPID']
varnames = ['tt', 'tc', 'ACT_PT', 'WT_PT', 'EMP_DENS', 'ADUL_VEH']
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

def df_coeff_col(seed,dataframe,names_asvars,choiceset,var_alt):
    np.random.seed(seed)
    random_matrix = np.random.randint(1,len(choiceset)+1,(len(choiceset),len(names_asvars)))
    #print(random_matrix)
    
    ## Finding coefficients type (alt-specific or generic) for corresponding variables
    alt_spec_pos = []
    for i in range(random_matrix.shape[1]):
        pos_freq = pd.Series(range(len(random_matrix[:,i]))).groupby(random_matrix[:,i], sort=False).apply(list).tolist()
        alt_spec_pos.append(pos_freq)
    
    for i in range(len(alt_spec_pos)):
        for j in range(len(alt_spec_pos[i])):
            for k in range(len(alt_spec_pos[i][j])):
                alt_spec_pos[i][j][k] = choiceset[alt_spec_pos[i][j][k]]
    ## creating dummy columns based on the coefficient type
    asvars_new = []
    for i in range(len(alt_spec_pos)):
        for j in range(len(alt_spec_pos[i])):
            if len(alt_spec_pos[i][j]) < len(choiceset):
                dataframe[names_asvars[i] + '_' +'_'.join(alt_spec_pos[i][j])] = dataframe[names_asvars[i]] * np.isin(var_alt,alt_spec_pos[i][j])
                asvars_new.append(names_asvars[i] + '_' +'_'.join(alt_spec_pos[i][j]))
            else:
                asvars_new.append(names_asvars[i])
    return(asvars_new)    


new_asvars = df_coeff_col(1, df, varnames, choice_set, alt_var)

varnames = new_asvars

model = MultinomialLogit()




# init_coeff = [-2, -4, -3, -2, -1, -1, 0, 0, -0, -0.01, 0.0001, -1.15]
model.fit(X=df[varnames], y=choice_var, varnames=varnames, # init_coeff=np.repeat(.1, 11),
          isvars=[], alts=alt_var, ids=choice_id, # gtol=1e-1,
        #   randvars=randvars,
          fit_intercept=True,
        #   hess=False,
        #   gtol=1e-1,
        #   weights=[1, 1, 10, 10, 10, 100, 1]
          )#, init_coeff=init_coeff, tol=1e-2) #hess=False, grad=False)
model.summary()