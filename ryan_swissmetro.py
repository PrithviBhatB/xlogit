from xlogitprit import MixedLogit, MultinomialLogit

import pandas as pd
import numpy as np

df_wide = pd.read_table("http://transp-or.epfl.ch/data/swissmetro.dat", sep="\t")

# Keep only samples with known choice and purpose commute or business
df_wide = df_wide[(df_wide['PURPOSE'].isin([1, 3]) & (df_wide['CHOICE'] != 0))]
df_wide['custom_id'] = np.arange(len(df_wide))  # Add unique identifier column

# Rename columns to match format expected by pandas (first varname and then suffix)
df_wide.rename(columns={"TRAIN_TT": "time_train", "SM_TT": "time_sm", "CAR_TT": "time_car",
                        "TRAIN_CO": "cost_train","SM_CO": "cost_sm", "CAR_CO": "cost_car",
                        "TRAIN_HE": "headway_train", "SM_HE": "headway_sm", "SM_SEATS": "seatconf_sm",
                        "TRAIN_AV": "av_train", "SM_AV": "av_sm","CAR_AV": "av_car"}, inplace=True)
 
# Convert from wide to long format using pandas.
df = pd.wide_to_long(df_wide, ["time", "cost", "headway", "seatconf", "av"],
                     i="custom_id", j="alt", sep="_", suffix='\w+').sort_values(
                         by=['custom_id', 'alt']).reset_index()

# Fill unexisting values for some alternatives
df = df.fillna(0)  
# Format the outcome variable approapriatly
df["CHOICE"] = df["CHOICE"].map({1: 'train', 2:'sm', 3: 'car'})
# Convert CHOICE to True if alternative was selected; False otherwise
df["CHOICE"] = df["CHOICE"] == df["alt"]
# Scale variables
df['time'] = df['time']/100
train_pass = ((df["GA"] == 1) & (df["alt"].isin(['train', 'sm']))).astype(int)
df['cost'] = df['cost']*(train_pass==0)/100
 
# Create alternative specific constants
df['asc_train'] = np.ones(len(df))*(df['alt'] == 'train')
df['asc_car'] = np.ones(len(df))*(df['alt'] == 'car')

varnames = ['asc_car', 'asc_train', 'cost', 'time', 'luggage_car',
            'he_sm_train', 'seats', 'ga_sm_train', 'age_train']
model = MultinomialLogit()
model.fit(X=df[varnames], y=df['CHOICE'], varnames=varnames, alts=df['alt'])
# model = MixedLogit()
# model.fit(X=df[varnames], y=df['CHOICE'], varnames=varnames, alts=df['alt'],
#         #   transvars=['cost'],
#           ids=df['custom_id'], avail=df['av'], randvars={'time': 'n'}, n_draws=2000,
#           # tol=1e-10
#           )
model.summary()