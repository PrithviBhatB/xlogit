import pandas as pd
import numpy as np
from xlogitprit import MultinomialLogit

df = pd.read_csv("https://raw.githubusercontent.com/arteagac/xlogit/master/examples/data/electricity_long.csv")

varnames = ["pf", "cl", "loc", "wk", "tod", "seas"]
X = df[varnames].values
y = df['choice'].values
choice_id = df['chid']
alt = [1, 2, 3, 4]
np.random.seed(123)


model = MultinomialLogit()
model.fit(
    X=df[varnames],
    y=y,
    varnames=varnames,
    isvars=[],
    alts=alt,
    fit_intercept=True,
    # hess=False,
    # grad=False,
    # method="L-BFGS-B"
    # tol=1e-4,
    # scipy_optimisation=True
)

model.summary()