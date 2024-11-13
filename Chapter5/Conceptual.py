# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.4
#   kernelspec:
#     display_name: islpenv
#     language: python
#     name: islpenv
# ---

# %% [markdown]
# # Conceptual Exercises

# %% [markdown]
# # Import notebook functions

# %%
from notebookfuncs import *

# %% [markdown]
# ## Exercise 1
#
# **Using basic statistical properties of the variance, as well as single-variable calculus, derive
# $$
# \large \alpha = \frac {\sigma_Y^2 - \sigma_{XY}^2 } {\sigma_X^2 + \sigma_Y^2 - 2\sigma_{XY}^2 }
# $$.**
#
# **In other words, prove that $\alpha$ given by the equation above does indeed minimize $Var(\alpha X + (1 − \alpha)Y)$.**

# %% [markdown]
# ## Exercise 2
#
# **We will now derive the probability that a given observation is part of a bootstrap sample. Suppose that we obtain a bootstrap sample from a set of n observations.**

# %% [markdown]
# ### (a)
# **What is the probability that the first bootstrap observation is not the $j_{th}$ observation from the original sample? Justify your answer.**

# %% [markdown]
# ### (b) 
# **What is the probability that the second bootstrap observation is not the $j_{th}$ observation from the original sample?**

# %% [markdown]
# ### (c) 
# **Argue that the probability that the $j_{th}$ observation is not in the bootstrap sample is $(1 − \frac {1} {n})^n$.**

# %% [markdown]
# ### (d) 
# **When n = 5, what is the probability that the $j_{th}$ observation is in the bootstrap sample?**

# %% [markdown]
# ### (e) 
# **When n = 100, what is the probability that the $j_{th}$ observation is in the bootstrap sample?**

# %% [markdown]
# ### (f)
# **When n = 10, 000, what is the probability that the $j_{th}$ observation is in the bootstrap sample?**

# %% [markdown]
# ### (g) 
# **Create a plot that displays, for each integer value of n from 1 to 100, 000, the probability that the $j_{th}$ observation is in the bootstrap sample. Comment on what you observe.**

# %% [markdown]
# ### (h)
# **We will now investigate numerically the probability that a bootstrap sample of size n = 100 contains the $j_{th}$ observation. Here j = 4. We first create an array store with values that will subsequently be overwritten using the function `np.empty()`. We then repeatedly create bootstrap samples, and each time we record whether or not the fifth observation is contained in the bootstrap sample.**
#
# ```python
# rng = np.random.default_rng (10)
# store = np.empty (10000)
# for i in range (10000):
#   store[i] = np.sum(rng.choice (100, replace=True) == 4) > 0
# np.mean(store)
# ```
# **Comment on the results obtained.**

# %% [markdown]
# ## Exercise 3
# **We now review k-fold cross-validation.**

# %% [markdown]
# ### (a) 
# **Explain how k-fold cross-validation is implemented.**

# %% [markdown]
# ### (b) 
# **What are the advantages and disadvantages of k-fold cross-validation relative to:**
#
#  **i.  The validation set approach?**
# **ii.  LOOCV?**

# %% [markdown]
# ## Exercise 4
# **Suppose that we use some statistical learning method to make a prediction for the response Y for a particular value of the predictor X. Carefully describe how we might estimate the standard deviation of our prediction.**

# %%
allDone();
