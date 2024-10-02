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
# # 11) In this problem we will investigate the t-statistic for the null hypothesis H0 : $\beta$ = 0 in simple linear regression without an intercept. To begin, we generate a predictor x and a response y as follows.

# %% [markdown]
# ```python
# rng = np.random.default_rng (1)
# x = rng.normal(size =100)
# y = 2 * x + rng.normal(size =100)
# ```

# %% [markdown]
# ## (a) Perform a simple linear regression of y onto x, without an intercept. Report the coefficient estimate $\hat{\beta}$ the standard error of this coefficient estimate, and the t-statistic and p-value associated with the null hypothesis H0 : $\beta$ = 0. Comment on these results. (You can perform regression without an intercept using the keywords argument intercept=False to ModelSpec().)

# %% [markdown]
# ## (b) Now perform a simple linear regression of x onto y without an intercept, and report the coefficient estimate, its standard error, and the corresponding t-statistic and p-values associated with the null hypothesis H0 : $\beta$ = 0. Comment on these results.

# %% [markdown]
# ## (c) What is the relationship between the results obtained in (a) and (b)?

# %% [markdown]
# ## (d) For the regression of Y onto X without an intercept, the t-statistic for H0 : β = 0 takes the form $\hat{\beta} / SE(\hat{\beta})$, where $\hat{\beta}$ is given by (3.38), and where 

# %% [markdown]
# $$
# \Large SE(\hat{\beta}) = \sqrt{\frac {\sum_{i=1}^{n}(y_{i} - x_{i}\hat{\beta})^{2}} {(n - 1)\sum_{i^{'}=1}^n x_{i^{'}}^2}}
# $$
#

# %% [markdown]
# ## (These formulas are slightly different from those given in Sections 3.1.1 and 3.1.2, since here we are performing regression without an intercept.) Show algebraically, and confirm numerically in Python, that the t-statistic can be written as

# %% [markdown]
# $$
# \Large \frac {(n - 1) \sum_{i=1}^n x_{i}y_{i}} {\sqrt {(\sum_{i=1}^n x_{i}^2)(\sum_{i=1}^n y_{i}^2) - (\sum_{i^{'}=1}^{n}x_{i^{'}}y_{i^{'}})^{2}}}
# $$

# %% [markdown]
# ## (e) Using the results from (d), argue that the t-statistic for the regression of y onto x is the same as the t-statistic for the regression of x onto y. 

# %% [markdown]
# ## (f) In Python, show that when regression is performed with an intercept, the t-statistic for H0 : $\beta$1 = 0 is the same for the regression of y onto x as it is for the regression of x onto y.
