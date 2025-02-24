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

# %% [raw]
# {{< include codewraplatex.yml >}}

# %% [markdown]
# # Exercise 12

# %% [markdown]
# ## Import notebook funcs

# %%
from notebookfuncs import *

# %% [markdown]
# ## 12. This problem involves simple linear regression without an intercept.

# %% [markdown]
# ### (a) Recall that the coefficient estimate β̂ for the linear regression of Y onto X without an intercept is given by (3.38). Under what circumstance is the coefficient estimate for the regression of X onto Y the same as the coefficient estimate for the regression of Y onto X?

# %% [markdown]
# ### (b) Generate an example in Python with n = 100 observations in which the coefficient estimate for the regression of X onto Y is different from the coefficient estimate for the regression of Y onto X.

# %% [markdown]
# ### (c) Generate an example in Python with n = 100 observations in which the coefficient estimate for the regression of X onto Y is the same as the coefficient estimate for the regression of Y onto X.

# %% [markdown]
# - This has already been proved and shown in my answer to Exercise 11 where the coefficients are calculated as $\rho * \frac {SD(y)} {SD(x)}$ and its inverse.

# %% [markdown]
# - The ratios of the standard deviations are inversed when the regressions are inversed.

# %% [markdown]
# - When the two variables are standardizebed and have unit variance or SD, then their coefficient estimate $\hat{\beta}$ are the same as the Pearson correlation coefficient $\rho$.

# %% [markdown]
# #### Examples have been generated for the same in Exercise 11. 

# %%
allDone();
