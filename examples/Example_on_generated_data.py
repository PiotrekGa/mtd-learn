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

import numpy as np
from mtdlearn.mtd import MTD
from mtdlearn.preprocessing import PathEncoder
from mtdlearn.datasets import data_values3_order2_full
from sklearn.metrics import accuracy_score

x = data_values3_order2_full['x']
y = data_values3_order2_full['y']
sample_weight = data_values3_order2_full['sample_weight']

x[:5]

y[:5]

sample_weight[:5]

pe = PathEncoder(3)
pe.fit(x, y)

pe.label_dict

x_tr3, y_tr3 = pe.transform(x, y)

x_tr3[:5]

y_tr3[:5]

model = MTD(n_dimensions=3, order=3)

model.fit(x_tr3, y_tr3)

model.bic

model.predict(pe.transform(np.array([['A>B>C'], ['B>B>A'], ['C>C>C']])))

model.predict_proba(pe.transform(np.array([['A>B>C'], ['B>B>A'], ['C>C>C']])))
