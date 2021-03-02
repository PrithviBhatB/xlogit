import pandas as pd
import numpy as np
from xlogitprit import MultinomialLogit

df_wide = pd.read_csv("examples/data/swissmetro_training.csv")
df_wide['custom_id'] = np.arange(len(df_wide))  # Add unique identifier

#Let's rename some columns for convenient reshaping using pandas
df_wide.rename(columns={"TRAIN_TT": "time_train", "SM_TT": "time_sm", "CAR_TT": "time_car",
                        "TRAIN_CO": "cost_train","SM_CO": "cost_sm", "CAR_CO": "cost_car",
                        "TRAIN_HE": "headway_train", "SM_HE": "headway_sm", "SM_SEATS": "seatconf_sm",
                        "TRAIN_AV": "av_train", "SM_AV": "av_sm","CAR_AV": "av_car"}, inplace=True)
 
# Convert from wide to long format using pandas.
df = pd.wide_to_long(df_wide, ["time", "cost", "headway", "seatconf", "av"],
                     i="custom_id", j="alt", sep="_", suffix='\w+').sort_values(
                         by=['custom_id', 'alt']).reset_index()
df = df.fillna(0)  # Fill unexisting values for some alternatives

# Format the outcome variable approapriately
df["CHOICE"] = df["CHOICE"].map({1: 'train', 2: 'sm', 3: 'car'})
# Convert CHOICE to True if alternative was selected; False otherwise
df["CHOICE"] = df["CHOICE"] == df["alt"]

# Create model specification
# Alternative Specific Constants
df['asc_train'] = np.ones(len(df))*(df['alt'] == 'train')
df['asc_car'] = np.ones(len(df))*(df['alt'] == 'car')

# Coefficient GA for swissmetro and train
df['ga_sm_train'] = df['GA']*((df['alt'] == 'train')|(df['alt'] == 'sm'))

# Coefficient headway for swissmetro and train
df['he_sm_train'] = df['headway']*((df['alt'] == 'train')|(df['alt'] == 'sm'))

# Coefficient Age for train
df['age_train'] = df['AGE']*(df['alt'] == 'train')

# Coefficient Luggage for car
df['luggage_car'] = df['LUGGAGE']*(df['alt'] == 'car')

# Coefficient seatsconfig for car
df['seats'] = df['seatconf']*(df['alt'] == 'sm')

varnames = ['asc_train', 'asc_car', 'cost', 'time', 'luggage_car',
            'he_sm_train', 'seats', 'ga_sm_train', 'age_train']
model = MultinomialLogit()
model.fit(X=df[varnames],
          y=df['CHOICE'],
          varnames=varnames,
          alts=df['alt'],
          ids=df['custom_id'],
          avail=df['av'],
          # transvars=['luggage_car']
        #   init_coeff=np.random.normal(0, 1, 9)
        #   scipy_optimisation=False
        # method="L-BFGS-B"
        #   tol=1e-3
          )
model.summary()