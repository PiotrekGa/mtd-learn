---
# 2020-07-06 Mixture Transition Distribution models
_Model explanation and usage example_

## Intention

The article intends to outline the Mixture Transition Distribution models.
A reader will find here definitions of Markov Chain and MTD models and presentation of a Python package 
[mtd-learn](https://github.com/PiotrekGa/mtd-learn) for estimating them. This post does not intend to be 
exhaustive on the subject. References to more detailed sources will be listed at the end of the post.

Note to R users. There is an R package [march](https://cran.r-project.org/web/packages/march/) developed and
maintained by Andre Berchtold.

## Introduction

The Mixture Transition Distribution (MTD) model was proposed by Raftery in 1985<sup>[1]</sup>. Its initial intent 
was to approximate high order Markov Chains (MC), but it can serve as an independent model too. The main advantage of 
the MTD model is that its number of independent parameters of the MTD model grows linearly with the order in contrast to
the exponential growth of Markov Chains models.


### Model definition

#### Markov Chains recap

#### First-order Markov Chains

First-order Markov Chain is a sequence of random variables _(X<sub>n</sub>)_ such that:

<p align="center">
  <img src="https://piotrekga.github.io/images/CodeCogsEqn.png">
</p>


such that:

<p align="center">
  <img src="https://piotrekga.github.io/images/CodeCogsEqn1.png">
</p>

for every _i<sub>t</sub>...i<sub>0</sub> ∈ S_, where _S_ is a state space.

The probability can be written in an abbreviated form:

<p align="center">
  <img src="https://piotrekga.github.io/images/CodeCogsEqn3.png">
</p>

If we assume that for every _t_:

<p align="center">
  <img src="https://piotrekga.github.io/images/CodeCogsEqn4.png">
</p>

we obtain time-homogeneous Markov Chain.

A transition matrix of a first-order Markov Chain is build of all possible combination of process states _i<sub>0</sub>_
and _i<sub>1</sub>_: 

<p align="center">
  <img src="https://piotrekga.github.io/images/CodeCogsEqn5.png">
</p>

_Q_ is a stochastic matrix. It means that all its values are [0, 1] and the sum of every row equals 1.

#### High-order Markov Chains

An _l_-order Markov Chain is a stochastic process in which current state depends on _l_ last observations:

<p align="center">
  <img src="https://piotrekga.github.io/images/CodeCogsEqn6.png">
</p>

and

<p align="center">
  <img src="https://piotrekga.github.io/images/CodeCogsEqn7.png">
</p>

such that:

<p align="center">
  <img src="https://piotrekga.github.io/images/CodeCogsEqn8.png">
</p>

for every _i<sub>t</sub>...i<sub>0</sub> ∈ S_, where _S_ is a state space.


It is possible to represent a high-order Markov Chain as a first-order Markov Chain. Some probabilities in the 
transition matrix _Q_ will be equal to zero by definition (structural zeros). 
Below is an example od 2-order MC with 3 possible states:

<p align="center">
  <img src="https://piotrekga.github.io/images/CodeCogsEqn9.png">
</p>

where

<p align="center">
  <img src="https://piotrekga.github.io/images/CodeCogsEqn10.png">
</p>

If we remove structural zeros from the _Q_ matrix we obtain its reduced form:

<p align="center">
  <img src="https://piotrekga.github.io/images/CodeCogsEqn11.png">
</p>

Please note, that the representation of _Q_ and _R_ is slightly different than the one from the original 
paper<sup>[2]</sup>. It's due to the implementation convenience of the Python package `mtd-learn`.

The number of independent parameters of high-order Markov Chain is equal to _m<sup>l</sup>(m-1)_, where _m_ represents
the number of states and _l_ is the order of the model.

The log-likelihood of the Markov Chain model is given by the formula:

<p align="center">
  <img src="https://piotrekga.github.io/images/CodeCogsEqn19.png">
</p>

where _n<sub>i<sub>l</sub>...i<sub>0</sub></sub>_ means number of transitions of type 
_X<sub>t-l</sub> = i<sub>l</sub>,...,X<sub>t-1</sub> = i<sub>1</sub>,X<sub>t</sub> = i<sub>0</sub>_ in a dataset.

The maximum likelihood estimator of a transition _q<sub>i<sub>l</sub>...i<sub>0</sub></sub>_ is:

<p align="center">
  <img src="https://piotrekga.github.io/images/CodeCogsEqn20.png">
</p>

where 

<p align="center">
  <img src="https://piotrekga.github.io/images/CodeCogsEqn21.png">
</p>

### MTD models

#### Mixture Transition  Distribution model

The MTD model is a sequence of random variables _(X<sub>n</sub>)_ such that:

<p align="center">
  <img src="https://piotrekga.github.io/images/CodeCogsEqn12.png">
</p>

where _i<sub>t</sub>...i<sub>0</sub> ∈ N_, probabilities _q<sub>i<sub>l</sub>i<sub>0</sub></sub>_ are elements of a
_m ⨯ m Q_ matrix and _𝜆  = (𝜆 <sub>l</sub>,...,𝜆 <sub>1</sub>)<sup>T</sup>_ is a weight vector.

Following conditions has to be met for the model to produce probabilities:

<p align="center">
  <img src="https://piotrekga.github.io/images/CodeCogsEqn13.png">
</p>


<p align="center">
  <img src="https://piotrekga.github.io/images/CodeCogsEqn14.png">
</p>


<p align="center">
  <img src="https://piotrekga.github.io/images/CodeCogsEqn15.png">
</p>

The log-likelihood function of the MTD model is given by:

<p align="center">
  <img src="https://piotrekga.github.io/images/CodeCogsEqn22.png">
</p>

where _n<sub>i<sub>l</sub>...i<sub>0</sub></sub>_ means number of transitions of type 
_X<sub>t-l</sub> = i<sub>l</sub>,...,X<sub>t-1</sub> = i<sub>1</sub>,X<sub>t</sub> = i<sub>0</sub>_ in a dataset.

#### Generalized Mixture Transition Distribution model

The MTDg model is a sequence of random variables _(X<sub>n</sub>)_ such that:

<p align="center">
  <img src="https://piotrekga.github.io/images/CodeCogsEqn16.png">
</p>

where _i<sub>t</sub>...i<sub>0</sub> ∈ N_, 𝜆  = (𝜆 <sub>l</sub>,...,𝜆 <sub>1</sub>)<sup>T</sup>_ is a weight vector and
_Q<sub>g</sub> = [q<sup>(g)</sup><sub>i<sub>g</sub>i<sub>0</sub></sub>]_ is an _m ⨯ m_ matrix representing the association
between _g_ lag and the current state.

Following conditions has to be met for the model to produce probabilities:

<p align="center">
  <img src="https://piotrekga.github.io/images/CodeCogsEqn17.png">
</p>


<p align="center">
  <img src="https://piotrekga.github.io/images/CodeCogsEqn18.png">
</p>

The log-likelihood function of the MTDg model is given by:

<p align="center">
  <img src="https://piotrekga.github.io/images/CodeCogsEqn23.png">
</p>

where _n<sub>i<sub>l</sub>...i<sub>0</sub></sub>_ means number of transitions of type 
_X<sub>t-l</sub> = i<sub>l</sub>,...,X<sub>t-1</sub> = i<sub>1</sub>,X<sub>t</sub> = i<sub>0</sub>_ in a dataset.

### MTD models intuition

You can think about the MTDg model as a weighted average of transition probabilities from subsequent lags. The example below 
shows how to calculate a probability of transition B->C->A->B from an order 3 MTDg model:

<p align="center">
  <img src="https://piotrekga.github.io/images/mtd.png">
</p>

In case of the MTD model all the Q<sup>(1)</sup>, Q<sup>(2)</sup>, and Q<sup>(3)</sup> matrices would be the same.

### Number of independent parameters

According to [1] the number independent parameters of the MTDg model equals _lm(m-1) + l - 1_. In [2] Lebre and 
Bourguignon proved that the true number of independent parameters equals _(ml - m + 1)(l - 1)_. Since the `mtd-learn`
package uses the estimation method proposed in [2] the number of parameters is calculated with the latest formula.

For Markov Chain the number of parameters equals _m<sup>l</sup>(m-1)_. 
You can find a comparison of the number of parameters below:

| States   |      Order    | Markov Chain | MTDg<sup>[1]</sup>  | MTDg|
|----------|:-------------:|-------------:|--------------------:|----:|
| 2        | 1             |     2        | 2                   | 2   |
| 2        | 2             |     4        | 3                   | 5   |
| 2        | 3             |     8        | 4                   | 8   |
| 2        | 4             |    16        | 5                   | 11  |
| 3        | 1             |     6        | 6                   | 6   |
| 3        | 2             |    18        | 10                  | 13  |
| 3        | 3             |    54        | 14                  | 20  |
| 3        | 4             |   162        | 18                  | 27  |
| 5        | 1             |    20        | 20                  | 20  |
| 5        | 2             |   100        | 36                  | 41  |
| 5        | 3             |   500        | 52                  | 62  |
| 5        | 4             |  2500        | 68                  | 83  |
| 10       | 1             |    90        | 90                  | 90  |
| 10       | 2             |   900        | 171                 | 181 |
| 10       | 3             |  9000        | 252                 | 272 |
| 10       | 4             | 90000        | 333                 | 363 |

## Information criteria

To determine the order of the MTD / MTDg model you can use one of the two information criteria - 
Akaike's and Bayesian (also known as Schwarz's):

<p align="center">
  <img src="https://piotrekga.github.io/images/CodeCogsEqn24.png">
</p>

<p align="center">
  <img src="https://piotrekga.github.io/images/CodeCogsEqn25.png">
</p>

The first part of each equation is the model's log-likelihood _ln(L)_. The second one is a penalty for the number of
independent parameters. In the case of BIC, the number of samples is also taken into consideration.

In the `mtd-learn` package you can access then with `MTD.aic` and `MTD.bic` properties. 
You should choose a model with the minimal value of the chosen criterion.

## Implementation details

### Estimation algorithm

You can find the Python implementation of the model here: [mtd-learn](https://github.com/PiotrekGa/mtd-learn). The
models are estimated using a version of the Expectation-Maximization algorithm proposed by Lebre and Bourguignon in [2]. 
In my master's thesis
I've checked that it yielded similar results, but was much faster and easier to implement, than the method proposed
by Berchtold in [3]. The explanation of the EM algorithm is out of the scope of the post. For details please refer to the 
original article.

### Output of the model

#### Transition matrices and lambda vector

The `mtd-learn` package returns _Q_ matrices and the 𝜆 vector in the following order:
```
mtd.transition_matrices.round(3) = array([[[0.521, 0.479], # Q3
                                           [0.404, 0.596]],
                                   
                                          [[0.254, 0.746], # Q2
                                           [0.569, 0.431]],
                                   
                                          [[0.797, 0.203], # Q1
                                           [0.542, 0.458]]])

#                            lambda3, lambda2, lambda1
mtd.lambdas.round(3) = array([0.082,   0.122,   0.796])

```

#### Reconstructed MC matrices

Based on the MTDg model it's possible to construct an MC transition matrix. The matrix looks like follows:

```
mtd.transition_matrix.round(3) = array([[0.708, 0.292],
                                        [0.505, 0.495],
                                        [0.746, 0.254],
                                        [0.543, 0.457],
                                        [0.698, 0.302],
                                        [0.495, 0.505],
                                        [0.737, 0.263],
                                        [0.534, 0.466]])
```

The expanded version of the matrix (simulating first-order MC from third-order MC) looks as follows:

```
mtd.expanded_matrix.round(3) = array([[0.708, 0.292, 0.   , 0.   , 0.   , 0.   , 0.   , 0.   ],
                                      [0.   , 0.   , 0.505, 0.495, 0.   , 0.   , 0.   , 0.   ],
                                      [0.   , 0.   , 0.   , 0.   , 0.746, 0.254, 0.   , 0.   ],
                                      [0.   , 0.   , 0.   , 0.   , 0.   , 0.   , 0.543, 0.457],
                                      [0.698, 0.302, 0.   , 0.   , 0.   , 0.   , 0.   , 0.   ],
                                      [0.   , 0.   , 0.495, 0.505, 0.   , 0.   , 0.   , 0.   ],
                                      [0.   , 0.   , 0.   , 0.   , 0.737, 0.263, 0.   , 0.   ],
                                      [0.   , 0.   , 0.   , 0.   , 0.   , 0.   , 0.534, 0.466]])
```

With states indexes:

```
        t-3, t-2, t-1
IDX = [( 0,   0,   0 ),
       ( 0,   0,   1 ),
       ( 0,   1,   0 ),
       ( 0,   1,   1 ),
       ( 1,   0,   0 ),
       ( 1,   0,   1 ),
       ( 1,   1,   0 ),
       ( 1,   1,   1 )]
```

For example, if we would like to find a transition probabilities after `1->0->1`, we have to choose row 6 of 
the R matrix (`[0.495, 0.505]`) or the 3rd and 4th elements of the 6th row of the Q matrix 
(`[0.   , 0.   , 0.495, 0.505, 0.   , 0.   , 0.   , 0.   ]`).

## Usage example

Let's analyze change patterns in the exchange rate between US Dollars and Euro between 1999-01-05 and 2020-04-10. You
can find the dataset [here](https://github.com/PiotrekGa/mtd-learn/blob/master/examples/euro_usd.csv).
Since the MTDg model work on discrete states the changes were binned into four groups:

1. 0_BIG_DROP - more that 0.5% drop
2. 1_DROP - less than 0.5% drop
3. 2_RISE - less that 0.5% rise
4. 3_BIG_RISE - more that 0.5% rise

Let's start with imports:

```
import pandas as pd
import numpy as np

from mtdlearn.mtd import MTD
from mtdlearn.preprocessing import PathEncoder, SequenceCutter
```

and follow with grouping code: 
```
df = pd.read_csv('euro_usd.csv')
df['Change'] = df.Closing_rate.diff()
df['Change_enc'] = np.nan
df.loc[df.Change < 0.0, 'Change_enc'] = '1_DROP'
df.loc[df.Change < -0.005, 'Change_enc'] = '0_BIG_DROP'
df.loc[df.Change >= 0, 'Change_enc'] = '2_RISE'
df.loc[df.Change >= 0.005, 'Change_enc'] = '3_BIG_RISE'
df.dropna(inplace=True)

df.Change_enc

1       0_BIG_DROP
2       0_BIG_DROP
3       3_BIG_RISE
4       0_BIG_DROP
5       0_BIG_DROP
           ...    
5516        1_DROP
5517    3_BIG_RISE
5518        1_DROP
5519    3_BIG_RISE
5520        2_RISE
```

Now we need to transform the `pd.Series` into a more `mtd-learn`-friendly format. You can use the `SequenceCutter` class
to do it. We will start with `order=2`.

```
order = 2

sc = SequenceCutter(order)
x, y = sc.transform(df.Change_enc.values)

x
array([['0_BIG_DROP>0_BIG_DROP'],
       ['0_BIG_DROP>3_BIG_RISE'],
       ['3_BIG_RISE>0_BIG_DROP'],
       ...,
       ['1_DROP>3_BIG_RISE'],
       ['3_BIG_RISE>1_DROP'],
       ['1_DROP>3_BIG_RISE']], dtype='<U21')

y
array(['3_BIG_RISE', '0_BIG_DROP', '0_BIG_DROP', ..., '1_DROP',
       '3_BIG_RISE', '2_RISE'], dtype=object)
```

We can see here that each state (in vector `y`) has a two-element sequence assigned in `x`. For instance, two first 
changes `0_BIG_DROP` and `0_BIG_DROP` are followed by the `3_BIG_RISE` state.

The values need to be encoded into integers. We can do it with the `PathEncoder` class:

```
pe = PathEncoder(order)
pe.fit(x, y)
x_tr, y_tr = pe.transform(x, y)

x_tr
array([[0, 0],
       [0, 3],
       [3, 0],
       ...,
       [1, 3],
       [3, 1],
       [1, 3]])

y_tr
array([3, 0, 0, ..., 1, 3, 2])
```

You can access the encoding dictionary:

```
pe.label_dict
{'0_BIG_DROP': 0, '1_DROP': 1, '2_RISE': 2, '3_BIG_RISE': 3, 'null': 4}
```

Note that there is a `null` label dedicated to missing values.

To fit the model simply create the `MTD` class object and fit it:

```
model = MTD(order=order)
model.fit(x_tr, y_tr)
log-likelihood value: -7547.882973125838
```

You can check the values of information criteria:

```
print(f"AIC: {model.aic.round(1)}, BIC: {model.bic.round(1)}")
AIC: 15137.8, BIC: 15276.7
```

And make predictions:

```
model.predict(np.array([[0, 0], 
                        [1, 3]]))
array([2, 1])

model.predict_proba(np.array([[0, 0], 
                              [1, 3]])).round(3)
array([[0.239, 0.239, 0.306, 0.215],
       [0.217, 0.315, 0.275, 0.192]])
```

Let's run the whole code for `order=3`:

```
order = 3
​
sc = SequenceCutter(order)
x, y = sc.transform(df.Change_enc.values)

pe = PathEncoder(order)
pe.fit(x, y)
x_tr, y_tr = pe.transform(x, y)

model = MTD(order=order, n_jobs=-1, number_of_initiations=100)
model.fit(x_tr, y_tr)
print(f"AIC: {model.aic.round(1)}, BIC: {model.bic.round(1)}")

log-likelihood value: -7535.536495080953
AIC: 15131.1, BIC: 15329.5
```

The AIC shows we should choose `order=3`, but the BIC says `order=2`. As Segal's law states: "A man with a watch 
knows what time it is. A man with two watches is never sure.", so choose your criterion prior to checking its value :)

We can compare it with the performance of Markov Chains:

```
order = 2
model = MTD(order=order)
model.fit(x_tr, y_tr)
log-likelihood value: -7528.058152541998
AIC: 15152.1, BIC: 15469.7

order = 3
model = MarkovChain(order=order)
model.fit(x_tr, y_tr)

log-likelihood value: -7421.656650869228
AIC: 15227.3, BIC: 16497.5
```

No matter which criterion we choose here, the MTD models seem to generalize better than Markov Chains.

## Summary

The Mixture Transition Distribution models are a parsimonious alternative to finite states Markov Chains. Now the 
`mtd-learn` package offers an easy way to use them with Python. The package is in an early stage of development. Any
contribution or feature requests are welcome. You can find more resources regarding the models in the Bibliography
section below. 

LaTeX formulas were generated with [latex.codecogs.com](https://www.codecogs.com/latex/eqneditor.php)

## Bibliography
1. BERCHTOLD, RAFTERY, The mixture Transition Distribution Model for High-Order Markov Chains
and Non-Gaussian Time Series , 2002., Statistical Science Vol. 17, No. 3, 328-356
2. LEBRE, BOURGUIGNON, An EM algorithm for estimation in the Mixture Transition Distribution
model , Laboratoire Statistique et Genome, Universite Evry Val d'Essonne, Evry, 2009
3. BERCHTOLD, Estimation of the Mixture Transition Distribution Model . 1999, Technical Report no. 352, Department of 
Statistics, University of Washington
