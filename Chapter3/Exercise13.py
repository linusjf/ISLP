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
# # Exercise 13

# %% [markdown]
# ## Import notebook funcs

# %%
from notebookfuncs import *

# %% [markdown]
# ## In this exercise you will create some simulated data and will fit simple linear regression models to it. Make sure to use the default random number generator with seed set to 1 prior to starting part (a) to ensure consistent results.

# %% [markdown]
# ### (a) Using the normal() method of your random number generator, create a vector, x, containing 100 observations drawn from a N (0, 1) distribution. This represents a feature, X.

# %% [markdown]
# ### (b) Using the normal() method, create a vector, eps, containing 100 observations drawn from a N (0, 0.25) distribution—a normal distribution with mean zero and variance 0.25.

# %% [markdown]
# ### (c) Using x and eps, generate a vector y according to the model 
# #### $Y = −1 + 0.5 * X + \epsilon$

# %% [markdown]
# #### What is the length of the vector y? What are the values of $\beta_0$ and $\beta_1$ in this linear model?

# %% [markdown]
# ### (d) Create a scatterplot displaying the relationship between x and y. Comment on what you observe.

# %% [markdown]
# ### (e) Fit a least squares linear model to predict y using x. Comment on the model obtained. How do $\hat{\beta_0}$ and $\hat{\beta_1}$ compare to $\beta_0$ and $\beta_1$ ?

# %% [markdown]
# ### (f) Display the least squares line on the scatterplot obtained in (d). Draw the population regression line on the plot, in a different color. Use the legend() method of the axes to create an appropriate legend.

# %% [markdown]
# ### (g) Now fit a polynomial regression model that predicts $y$ using $x$ and $x^2$. Is there evidence that the quadratic term improves the model fit? Explain your answer.

# %% [markdown]
# ### (h) Repeat (a)–(f) after modifying the data generation process in such a way that there is less noise in the data. The model (3.39) should remain the same. You can do this by decreasing the variance of the normal distribution used to generate the error term " in (b). Describe your results.

# %% [markdown]
# ### (i) Repeat (a)–(f) after modifying the data generation process in such a way that there is more noise in the data. The model (3.39) should remain the same. You can do this by increasing the variance of the normal distribution used to generate the error term " in (b). Describe your results.

# %% [markdown]
# ### (j) What are the confidence intervals for β0 and β1 based on the original data set, the noisier data set, and the less noisy data set? Comment on your results.

# %%
allDone();
