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
# # Exercise 14

# %% [markdown]
# ## Import notebook funcs

# %%
from notebookfuncs import *

# %% [markdown]
# ## (a) Perform the following commands in Python:

# %% [markdown]
# ```python
# rng = np.random.default_rng (10)
# x1 = rng.uniform (0, 1, size =100)
# x2 = 0.5 * x1 + rng.normal(size =100) / 10
# y = 2 + 2 * x1 + 0.3 * x2 + rng.normal(size =100)
# ```
# *The last line corresponds to creating a linear model in which $y$ is a function of $x_1$ and $x_2$. Write out the form of the linear model. What are the regression coefficients?*

# %% [markdown]
# ## (b) What is the correlation between $x_1$ and $x_2$? Create a scatterplot displaying the relationship between the variables.

# %% [markdown]
# ## (c) Using this data, fit a least squares regression to predict y using $x_1$ and $x_2$. Describe the results obtained. What are $\beta_0$, $\beta_1$ , and $\beta_2$ ? How do these relate to the true $\beta_0$, $\beta_1$ , and $\beta_2$ ? Can you reject the null hypothesis $H_0 : \beta_1 = 0$? How about the null hypothesis $H_0 : \beta_2 = 0$?

# %% [markdown]
# ## (d) Now fit a least squares regression to predict y using only $x_1$. Comment on your results. Can you reject the null hypothesis $H_0 : \beta_1 = 0$?

# %% [markdown]
# ## (e) Now fit a least squares regression to predict y using only $x_2$. Comment on your results. Can you reject the null hypothesis $H_0 : \beta_1 = 0$?

# %% [markdown]
# ## (f) Do the results obtained in (c)–(e) contradict each other? Explain your answer.

# %% [markdown]
# ## (g) Suppose we obtain one additional observation, which was unfortunately mismeasured. We use the function np.concatenate() to add this additional observation to each of $x_1, x_2 \: and \: y$.
# ```python
# x1 = np.concatenate ([x1 , [0.1]])
# x2 = np.concatenate ([x2 , [0.8]])
# y = np.concatenate ([y, [6]])
# ```
# Re-fit the linear models from (c) to (e) using this new data. What effect does this new observation have on the each of the models? In each model, is this observation an outlier? A high-leverage point? Both? Explain your answers.
#

# %%
allDone();
