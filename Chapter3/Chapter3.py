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

# %% [markdown]
# ## Using Transformations: Fit and Transform

# %%
design = MS(["lstat"])
design = design.fit(Boston)
X = design.transform(Boston)
X.head()

# %%
design = MS(["lstat"])
design = design.fit_transform(Boston)
X.head()

# %% [markdown]
# Full and exhaustive summary of the fit

# %%
results.summary()

# %% [markdown]
# Fitted coefficients can be retrieved as the *params* attribute of results

# %%
results.params

# %% [markdown]
# ### Computing predictions

# %%
design = MS(["lstat"])
new_df = pd.DataFrame({"lstat": [5,10,15]})
print(new_df)
newX = design.fit_transform(new_df)
newX

# %%
new_predictions = results.get_prediction(newX)
new_predictions.predicted_mean

# %%
new_predictions.conf_int(alpha=0.05)

# %%
new_predictions.conf_int(obs=True,alpha=0.05)

# %%
