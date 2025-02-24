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
# # Applied: Exercise 14

# %% [markdown]
# ## Import notebook funcs

# %%
from notebookfuncs import *

# %% [markdown]
# ## Exercise 14

# %% [markdown]
# In this problem, you will develop a model to predict whether a given car gets high or low gas mileage based on the Auto data set.

# %% [markdown]
# ## (a)

# %% [markdown]
# Create a binary variable, mpg01, that contains a 1 if mpg contains a value above its median, and a 0 if mpg contains a value below its median. You can compute the median using the median() function. Note you may find it helpful to use the data.frame() function to create a single data set containing both mpg01 and the other Auto variables.

# %% [markdown]
#  ### (b)

# %% [markdown]
# Explore the data graphically in order to investigate the association between mpg01 and the other features. Which of the other features seem most likely to be useful in predicting mpg01? Scatter plots and box plots may be useful tools to answer this question. Describe your findings.

# %% [markdown]
# ### (c)

# %% [markdown]
# Split the data into a training set and a test set.

# %% [markdown]
# ### (d)

# %% [markdown]
# Perform LDA on the training data in order to predict mpg01 using the variables that seemed most associated with mpg01 in (b). What is the test error of the model obtained?

# %% [markdown]
# ### (e)

# %% [markdown]
# Perform QDA on the training data in order to predict mpg01 using the variables that seemed most associated with mpg01 in (b). What is the test error of the model obtained?

# %% [markdown]
# ### (f)

# %% [markdown]
# Perform logistic regression on the training data in order to predict mpg01 using the variables that seemed most associated with mpg01 in (b). What is the test error of the model obtained?

# %% [markdown]
# ### (g)

# %% [markdown]
# Perform naive Bayes on the training data in order to predict mpg01 using the variables that seemed most associated with mpg01 in (b). What is the test error of the model obtained?

# %% [markdown]
# ### (h)

# %% [markdown]
# Perform KNN on the training data, with several values of K, in order to predict mpg01. Use only the variables that seemed most associated with mpg01 in (b). What test errors do you obtain? Which value of K seems to perform the best on this data set?

# %%
allDone();
