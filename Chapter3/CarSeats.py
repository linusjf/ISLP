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
# # Multilinear Regression: CarSeats dataset

# %% [markdown]
# ## Import notebook funcs

# %%
from notebookfuncs import *

# %% [markdown]
# ## Import standard libraries

# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.pyplot import subplots
import seaborn as sns
import itertools

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

# %% [markdown]
# ## Import User Funactions

# %%
from userfuncs import *

# %%
Carseats = load_data('Carseats')
Carseats.head()

# %%
Carseats.shape

# %%
Carseats = Carseats.dropna()
Carseats.shape

# %%
Carseats.describe()

# %%
Carseats["US"] = Carseats["US"].astype("category")
Carseats["Urban"] = Carseats["Urban"].astype("category")

# %% [markdown]
# ### (a) Fit a multiple regression model to predict Sales using Price, Urban, and US.

# %%
cols = list(Carseats.columns)
cols.remove("Sales")
formula = "Price + Urban + US"
perform_analysis("Sales", formula, Carseats);

# %% [markdown]
# ### (b) Provide an interpretation of each coefficient in the model. Be careful—some of the variables in the model are qualitative!

# %% [markdown]
# ### (c) Write out the model in equation form, being careful to handle the qualitative variables properly.

# %% [markdown]
# ### (d) For which of the predictors can you reject the null hypothesis H0 : βj = 0?

# %% [markdown]
# ### (e) On the basis of your response to the previous question, fit a smaller model that only uses the predictors for which there is evidence of association with the outcome.

# %% [markdown]
# ### (f) How well do the models in (a) and (e) fit the data?

# %% [markdown]
# ### (g) Using the model from (e), obtain 95 % confidence intervals for the coefficient(s).

# %% [markdown]
# ### (h) Is there evidence of outliers or high leverage observations in the model from (e)?

# %%
allDone()
