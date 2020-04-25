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
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from mtdlearn.mtd import MTD
from mtdlearn.preprocessing import PathEncoder
from mtdlearn.datasets import ChainGenerator

import warnings
warnings.filterwarnings("ignore")
# -

# ## Generate data

cg = ChainGenerator(('A', 'B', 'C'), 3, min_len=4, max_len=5)

x, y = cg.generate_data(1000)

x[:5]

y[:5]

# ## Encode paths

pe = PathEncoder(3)
pe.fit(x, y)

pe.label_dict

x_tr3, y_tr3 = pe.transform(x, y)

x_tr3[:5]

y_tr3[:5]

# ## Fitting model

model = MTD(order=3)

model.fit(x_tr3, y_tr3)

# ## Information criteria

model.aic

model.bic

# ## Trained parameters

model.lambdas.round(3)

model.transition_matrices.round(3)

sns.barplot(x=[f't - {abs(i-3)}' for i in range(model.order)], y=model.lambdas, palette='Reds');

# +
fig, axn = plt.subplots(1, 3, sharex=True, sharey=True, figsize=(12, 4))
cbar_ax = fig.add_axes([.91, .3, .03, .4])

for i, ax in enumerate(axn.flat):
    sns.heatmap(model.transition_matrices[i], ax=ax,
                cbar=i == 0,
                vmin=0, vmax=1,
                cmap='Reds',
                cbar_ax=None if i else cbar_ax)
    ax.set_title(f't - {3-i}')

fig.tight_layout(rect=[0, 0, .9, 1]);
# -

# ## Predict

model.predict(pe.transform(np.array([['A>B>C'], ['B>B>A'], ['C>C>C']])))

model.predict_proba(pe.transform(np.array([['A>B>C'], ['B>B>A'], ['C>C>C']])))
