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

# %% [markdown]
# ## Exercise 3 
# This problem relates to the QDA model, in which the observations within each class are drawn from a normal distribution with a class specific mean vector and a class specific covariance matrix. We consider the simple case where p = 1; i.e. there is only one feature. Suppose that we have K classes, and that if an observation belongs to the kth class then X comes from a one-dimensional normal distribution, $X ∼ N(\mu_k , \sigma_k^2)$. Recall that the density function for the one-dimensional normal distribution is given in 
# $$
# \huge f_k(x) = \frac {1} {\sqrt{2\pi}\sigma_k} exp(- \frac {1} {2\sigma_k^2}(x - \mu_k)^2)
# $$
#
# (4.16). Prove that in this case, the Bayes classifier is not linear. Argue that it is in fact quadratic. 
# *Hint: For this problem, you should follow the arguments laid out in Section 4.4.1, but without making the assumption that $\sigma_1^2 = · · · = \sigma_K^2$*

# %% [markdown]
# ## Exercise 4
# When the number of features p is large, there tends to be a deterioration in the performance of KNN and other local approaches that perform prediction using only observations that are near the test observation for which a prediction must be made. This phenomenon is known as the curse of dimensionality, and it ties into the fact that non-parametric approaches often perform poorly when p is large. We will now investigate this curse.

# %% [markdown]
# ### (a)
# Suppose that we have a set of observations, each with measurements on p = 1 feature, X. We assume that X is uniformly (evenly) distributed on [0, 1]. Associated with each observation is a response value. Suppose that we wish to predict a test observation’s response using only observations that are within 10 % of the range of X closest to that test observation. For instance, in order to predict the response for a test observation with X = 0.6, we will use observations in the range [0.55, 0.65]. On average, what fraction of the available observations will we use to make the prediction?

# %%
allDone();
