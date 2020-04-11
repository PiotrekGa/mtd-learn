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
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from mtdlearn.mtd import MTD
from mtdlearn.preprocessing import PathEncoder
from mtdlearn.datasets import data_values3_order2_full

import warnings
warnings.filterwarnings("ignore")
# -

# ## Import data

x = data_values3_order2_full['x']
y = data_values3_order2_full['y']
sample_weight = data_values3_order2_full['sample_weight']

x[:5]

y[:5]

sample_weight[:5]

# ## Encode paths

pe = PathEncoder(3)
pe.fit(x, y)

pe.label_dict

x_tr3, y_tr3 = pe.transform(x, y)

x_tr3[:5]

y_tr3[:5]

# ## Fitting model

model = MTD(n_dimensions=3, order=3)

model.fit(x_tr3, y_tr3)

# ## Information criteria

model.aic

model.bic

# ## Trained parameters

model.lambdas.round(3)

model.transition_matrices.round(3)

sns.barplot(x=[i for i in range(model.order)], y=model.lambdas, palette='Reds');

# +
fig, axn = plt.subplots(1, 3, sharex=True, sharey=True, figsize=(12, 4))
cbar_ax = fig.add_axes([.91, .3, .03, .4])

for i, ax in enumerate(axn.flat):
    sns.heatmap(model.transition_matrices[i], ax=ax,
                cbar=i == 0,
                vmin=0, vmax=1,
                cmap='Reds',
                cbar_ax=None if i else cbar_ax)
    ax.set_title(f'{3-i} lag')

fig.tight_layout(rect=[0, 0, .9, 1]);
# -

# ## Predict

model.predict(pe.transform(np.array([['A>B>C'], ['B>B>A'], ['C>C>C']])))

model.predict_proba(pe.transform(np.array([['A>B>C'], ['B>B>A'], ['C>C>C']])))
