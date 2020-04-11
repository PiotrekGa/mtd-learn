# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.3.3
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# +
import pandas as pd
import numpy as np
import seaborn as sns

from mtdlearn.mtd import MTD
from mtdlearn.preprocessing import PathEncoder

# +
df = pd.read_csv('euro_usd.csv')

df['Change'] = df.Closing_rate.diff()

df['Change_enc'] = np.nan

df.loc[df.Change < 0.0, 'Change_enc'] = '1_DROP'
df.loc[df.Change < -0.005, 'Change_enc'] = '0_BIG_DROP'
df.loc[df.Change >= 0, 'Change_enc'] = '2_RISE'
df.loc[df.Change >= 0.005, 'Change_enc'] = '3_BIG_RISE'


df['Change_lagged_1'] = df.Change_enc.shift(1)
df['Change_lagged_2'] = df.Change_enc.shift(2)
df['Change_lagged_3'] = df.Change_enc.shift(3)
df['Change_lagged_4'] = df.Change_enc.shift(4)

df.dropna(inplace=True)

x = df.Change_lagged_4 + '>' + df.Change_lagged_3 + '>' + df.Change_lagged_2 + '>' + df.Change_lagged_1
y = df.Change_enc

x = x.values.astype(str).reshape(-1, 1)
y = y.values.astype(str)

df.Change_enc.value_counts().sort_index()
# -

# ## Fit models

aics = []
bics = []

# +
order = 1
pe = PathEncoder(order)
pe.fit(x, y)

x_tr, y_tr = pe.transform(x, y)

model = MTD(n_dimensions=4, order=order, n_jobs=-1, number_of_initiations=100)
model.fit(x_tr, y_tr)
aics.append(model.aic)
bics.append(model.bic)

print(model.aic.round(1), model.bic.round(1))

# +
order = 2
pe = PathEncoder(order)
pe.fit(x, y)

x_tr, y_tr = pe.transform(x, y)

model = MTD(n_dimensions=4, order=order, n_jobs=-1, number_of_initiations=100)
model.fit(x_tr, y_tr)
aics.append(model.aic)
bics.append(model.bic)

print(model.aic.round(1), model.bic.round(1))

# +
order = 3
pe = PathEncoder(order)
pe.fit(x, y)

x_tr, y_tr = pe.transform(x, y)

model = MTD(n_dimensions=4, order=order, n_jobs=-1, number_of_initiations=100)
model.fit(x_tr, y_tr)
aics.append(model.aic)
bics.append(model.bic)

print(model.aic.round(1), model.bic.round(1))

# +
order = 4
pe = PathEncoder(order)
pe.fit(x, y)

x_tr, y_tr = pe.transform(x, y)

model = MTD(n_dimensions=4, order=order, n_jobs=-1, number_of_initiations=100)
model.fit(x_tr, y_tr)
aics.append(model.aic)
bics.append(model.bic)

print(model.aic.round(1), model.bic.round(1))
# -

# ## Choose model

sns.lineplot(x=[1, 2, 3, 4], y=aics)
sns.lineplot(x=[1, 2, 3, 4], y=bics);
