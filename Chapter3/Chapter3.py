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
# # Lab: Linear Regression

# %% [markdown]
# Import standard libraries

# %%
import numpy as np
import pandas as pd
from matplotlib.pyplot import subplots
import seaborn as sns

# %% [markdown]
# New imports

# %%
import statsmodels.api as sm

# %% [markdown]
# Import statsmodels objects

# %%
from statsmodels.stats.outliers_influence import variance_inflation_factor as VIF
from statsmodels.stats.anova import anova_lm

# %% [markdown]
# Import ISLP objects

# %%
from ISLP import load_data
from ISLP.models import (ModelSpec as MS, summarize, poly)

# %% [markdown]
# Inspecting objects and namespaces

# %%
dir()

# %%
A = np.array([3,5,11])
dir(A)

# %%
A.sum()

# %% [markdown]
# ## Simple Linear Regression

# %% [markdown]
# We will use the Boston housing dataset which is in the package ISLP

# %%
Boston = load_data("Boston")
Boston.columns

# %%
len(Boston.columns)

# %%
# Boston?

# %% [markdown]
# Use sm.OLS to fit a simple linear regression

# %%
X = pd.DataFrame({"intercept": np.ones(Boston.shape[0]),
                                       "lstat": Boston["lstat"]})
X.head()


# %% [markdown]
# Extract the response and fit the model.

# %%
y = Boston["medv"]
model = sm.OLS(y, X)
results = model.fit()

# %% [markdown]
# Summarize the results using the ISLP method summarize

# %%
summarize(results)

# %%
