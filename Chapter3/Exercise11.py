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
# # 11. In this problem we will investigate the t-statistic for the null hypothesis H0 : β = 0 in simple linear regression without an intercept. To begin, we generate a predictor x and a response y as follows.

# %% [markdown]
# ```python
# rng = np.random.default_rng (1)
# x = rng.normal(size =100)
# y = 2 * x + rng.normal(size =100)
# ```

# %% [markdown]
# ## (a) Perform a simple linear regression of y onto x, without an intercept. Report the coefficient estimate β̂, the standard error of this coefficient estimate, and the t-statistic and p-value associated with the null hypothesis H0 : β = 0. Comment on these results. (You can perform regression without an intercept using the keywords argument intercept=False to ModelSpec().)

# %% [markdown]
# ## (b) Now perform a simple linear regression of x onto y without an intercept, and report the coefficient estimate, its standard error, and the corresponding t-statistic and p-values associated with the null hypothesis H0 : β = 0. Comment on these results.

# %% [markdown]
# ## (c) What is the relationship between the results obtained in (a) and (b)?

# %% [markdown]
# ## (d) For the regression of Y onto X without an intercept, the t-statistic for H0 : β = 0 takes the form β̂/SE(β̂), where β̂ is given by (3.38), and where 
# SE(β̂) = 
