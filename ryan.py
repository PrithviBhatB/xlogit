from xlogitprit import MultinomialLogit

import pandas as pd
data_file = "https://raw.githubusercontent.com/arteagac/xlogit/master/examples/data/fishing_long.csv"
df = pd.read_csv(data_file)

varnames = ['price', 'catch', 'income']
X = df[varnames].values
y = df['choice'].values

model = MultinomialLogit()
model.fit(
  X,
  y,
  isvars=['income'],
  transvars=['price', 'catch'],
  transformation="boxcox",
  alts=['beach', 'boat', 'charter', 'pier'],
  varnames=varnames
)
model.summary()
