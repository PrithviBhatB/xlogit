from xlogit import MultinomialLogit

import pandas as pd
data_file = "https://raw.githubusercontent.com/arteagac/xlogit/master/examples/data/fishing_long.csv"
df = pd.read_csv(data_file)

varnames = ['income','price', 'catch']
X = df[varnames].values
y = df['choice'].values

model = MultinomialLogit()
model.fit(
  X,
  y,
  isvars = ['income'],
  transvars=['catch'],
  transformation="boxcox",
  alt=['beach','boat','charter','pier'],
  varnames= varnames
)
model.summary()
