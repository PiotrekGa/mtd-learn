# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.3.1
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# +
import pandas as pd
import numpy as np
import seaborn as sns

from mtdlearn.mtd import MTD, RandomWalk
from mtdlearn.preprocessing import PathEncoder, SequenceCutter

# +
df = pd.read_csv('euro_usd.csv')

df['Change'] = df.Closing_rate.diff()

df['Change_enc'] = np.nan

df.loc[df.Change < 0.0, 'Change_enc'] = '1_DROP'
df.loc[df.Change < -0.005, 'Change_enc'] = '0_BIG_DROP'
df.loc[df.Change >= 0, 'Change_enc'] = '2_RISE'
df.loc[df.Change >= 0.005, 'Change_enc'] = '3_BIG_RISE'

df.dropna(inplace=True)
# -

# ## Fit models

aics = []
bics = []

# +
order = 0

pe = PathEncoder(0, return_vector=True, input_vector=True)
y = pe.fit_transform(df.Change_enc.values.astype(str))

model = RandomWalk(4)
model.fit(y)

aics.append(model.aic)
bics.append(model.bic)

print(model.aic.round(1), model.bic.round(1))

# +
order = 1

sc = SequenceCutter(order)
x, y = sc.transform(df.Change_enc.values)

pe = PathEncoder(order)
pe.fit(x, y)

x_tr, y_tr = pe.transform(x, y)

model = MTD(order=order, n_jobs=-1, number_of_initiations=100)
model.fit(x_tr, y_tr)
aics.append(model.aic)
bics.append(model.bic)

print(model.aic.round(1), model.bic.round(1))

# +
order = 2

sc = SequenceCutter(order)
x, y = sc.transform(df.Change_enc.values)

pe = PathEncoder(order)
pe.fit(x, y)

x_tr, y_tr = pe.transform(x, y)

model = MTD(order=order, n_jobs=-1, number_of_initiations=100)
model.fit(x_tr, y_tr)
aics.append(model.aic)
bics.append(model.bic)

print(model.aic.round(1), model.bic.round(1))

# +
order = 3

sc = SequenceCutter(order)
x, y = sc.transform(df.Change_enc.values)

pe = PathEncoder(order)
pe.fit(x, y)

x_tr, y_tr = pe.transform(x, y)

model = MTD(order=order, n_jobs=-1, number_of_initiations=100)
model.fit(x_tr, y_tr)
aics.append(model.aic)
bics.append(model.bic)

print(model.aic.round(1), model.bic.round(1))

# +
order = 4

sc = SequenceCutter(order)
x, y = sc.transform(df.Change_enc.values)

pe = PathEncoder(order)
pe.fit(x, y)

x_tr, y_tr = pe.transform(x, y)

model = MTD(order=order, n_jobs=-1, number_of_initiations=100)
model.fit(x_tr, y_tr)
aics.append(model.aic)
bics.append(model.bic)

print(model.aic.round(1), model.bic.round(1))
# -

# ## Choose model

xs = [0, 1, 2, 3, 4]

sns.lineplot(x=xs, y=aics)
sns.lineplot(x=xs, y=bics);
