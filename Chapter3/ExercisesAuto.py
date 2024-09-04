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
# # Applied : Auto dataset

# %% [markdown]
# Import standard libraries

# %%
import numpy as np
import pandas as pd
from matplotlib.pyplot import subplots
#import seaborn as sns

# %% [markdown]
# New imports

# %%
import statsmodels.api as sm

# %% [markdown]
# Import statsmodel.objects

# %%
from statsmodels.stats.outliers_influence import variance_inflation_factor as VIF
from statsmodels.stats.outliers_influence import summary_table
from statsmodels.stats.anova import anova_lm

# %% [markdown]
# Import ISLP objects

# %%
import ISLP
from ISLP import models
from ISLP import load_data
from ISLP.models import (ModelSpec as MS, summarize, poly)

# %%
dir()

# %%
Auto = load_data('Auto')
Auto.columns

# %%
Auto.shape

# %%
Auto.describe()

# %% [markdown]
# ## Convert cylinders and origin columns to categorical types

# %%
Auto["cylinders"] = Auto["cylinders"].astype("category")
Auto["origin"] = Auto["origin"].astype("category")
Auto.describe()

# %% [markdown]
# ## 8) This question involves the use of Simple Linear Regression on the Auto dataset

# %% [markdown]
# ### (a) Use the sm.OLS() function to perform a simple linear regression with mpg as the response and horsepower as the predictor.
# ### Use the summarize() function to print the results.
# ### Comment on the output. For example:
#
# #### i. Is there a relationship between the predictor and the response?
#
# #### ii. How strong is the relationship between the predictor and the response?
#
# #### iii. Is the relationship between the predictor and the response positive or negative?
#
# #### iv. What is the predicted mpg associated with a horsepower of 98? What are the associated 95 % confidence and prediction intervals?

# %%
y = Auto["mpg"]
Auto = Auto.drop("mpg", axis=1)
Auto.head()

# %%
design = MS(["horsepower"])
design = design.fit(Auto)
X = design.transform(Auto)

# %%
model = sm.OLS(y, X)
results = model.fit()
summarize(results)

# %%
results.summary()

# %%
