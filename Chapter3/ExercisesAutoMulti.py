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
# # Multilinear Regression: Auto dataset

# %% [markdown]
# ## Import standard libraries

# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.pyplot import subplots

# %% [markdown]
# ## New imports

# %%
import statsmodels.api as sm

# %% [markdown]
# ## Import statsmodel.objects

# %%
from statsmodels.stats.outliers_influence import variance_inflation_factor as VIF
from statsmodels.stats.outliers_influence import summary_table
from statsmodels.stats.anova import anova_lm
import statsmodels.formula.api as smf

# %% [markdown]
# ## Import ISLP objects

# %%
import ISLP
from ISLP import models
from ISLP import load_data
from ISLP.models import (ModelSpec as MS, summarize, poly)

# %%
Auto = load_data('Auto')
Auto.columns

# %%
Auto.shape

# %%
Auto.describe()

# %% [markdown]
# ## 9. This question involves the use of multiple linear regression on the Auto data set.

# %% [markdown]
# ### (a) Produce a scatterplot matrix which includes all of the variables in the data set.

# %%
pd.plotting.scatter_matrix(Auto, figsize = (14, 14));

# %%
Auto.corr()

# %% [markdown]
# ## Convert cylinders and origin columns to categorical types

# %%
Auto["cylinders"] = Auto["cylinders"].astype("category")
Auto["origin"] = Auto["origin"].astype("category")
Auto.describe()

# %% [markdown]
# ### Encode categorical variables as dummy variables dropping the first to remove multicollinearity.

# %%
Auto = pd.get_dummies(Auto, columns=list(["cylinders", "origin"]), drop_first = True, dtype = np.uint8)
Auto.columns

# %%
y = Auto["mpg"]
Auto.columns.drop("mpg")

# %%
cols = list(Auto.columns)
cols.remove("mpg")
X = MS(cols).fit_transform(Auto)
formula = ' + '.join(cols)
model = smf.ols(f'mpg ~ {formula}', data=Auto)
results = model.fit()
summarize(results)

# %%
anova_lm(results)

# %%
