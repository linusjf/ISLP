# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.4
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Conceptual: Chapter 4 &mdash; Classification

# %% [raw]
# ## Import notebook functions

# %%
from notebookfuncs import *

# %% [markdown]
# ## Exercise 1
# Using a little bit of algebra, prove that
# $$
# \huge p(X)  = \frac {e^{\beta_0 + \beta_1 * X}} {1 + {e^{\beta_0 + \beta_1 * X}}}
# $$
#  is equivalent to
# $$
# \huge \frac {p(X)} {1 - p(X)} = e^{\beta_0 + \beta_1 * X}
# $$.
#
# In other words, the logistic function representation and logit representation for the logistic regression model are equivalent.

# %% [markdown]
# ## Exercise 2
# It was stated in the text that classifying an observation to the class for which
# $$
# \huge p_k(x) = \frac {\pi_k \frac {1} {\sqrt{2\pi}  \sigma} exp (- \frac {1} {2\sigma^2} (x - \mu_k)^2) } {\sum_{l=1}^K \pi_l \frac {1} {\sqrt{2\pi}  \sigma} exp (- \frac {1} {2\sigma^2} (x - \mu_l)^2)}
# $$
# (4.17)
# is largest is equivalent to classifying an observation to the class for which
# $$
# \huge \delta_k(x) = x.\frac {\mu_k} {\sigma^2} - \frac {\mu_k^2} {2\sigma^2} + log(\pi_k)
# $$
# (4.18) is the largest. Prove that this is the case. In other words, under the assumption that the observations in the $k_{th}$ class are drawn from a $N(\mu_k , \sigma^2)$ distribution, the Bayes classifier assigns an observation to the class for which the discriminant function is maximized.

# %%

# %%
allDone();
