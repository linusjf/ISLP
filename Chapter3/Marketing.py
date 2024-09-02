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
from statsmodels.stats.anova import anova_lm

# %% [markdown]
# Import ISLP objects

# %%
import ISLP
from ISLP import models
from ISLP import load_data
from ISLP.models import (ModelSpec as MS, summarize, poly)

# %% [markdown]
# Inspecting objects and namespaces

# %%
dir()

# %%
Advertising = pd.read_csv("Advertising.csv")
# Drop first column
Advertising = Advertising.iloc[:, 1:]
Advertising.head()

# %%
Advertising.describe()

# %% [markdown]
# ## Is there a relationship between sales and advertising budget?

# %%
y = Advertising["Sales"]
cols = list(Advertising.columns)
cols.remove("Sales")
X = MS(cols).fit_transform(Advertising)
model = sm.OLS(y, X)
results = model.fit()
print("F-value", results.fvalue)
print("F-pvalue", results.f_pvalue)
summarize(results)

# %%
dir(models)

# %% [markdown]
# The p-value corresponding to the F-statistic is very low. Thus, clear evidence of a relationship between sales and advertising budget.

# %%
dir(results)

# %%
## How strong is the relationship?

# %%
results.summary()

# %%
y.mean()

# %%
results.resid.std()

# %%
(results.resid.std() / y.mean()) * 100

# %% [markdown]
# The residual standard error (RSE) is 1.67 and the mean value of the response is 14.023 which translates to a percentage error of roughly 11.93%

# %%
("R-squared", results.rsquared, "Adjusted R-squared", results.rsquared_adj)

# %% [markdown]
# The R<sup>2</sup> explains about 90% of the variance in Sales.

# %% [markdown]
# ## Which media are associated with Sales?

# %% [markdown]
# The low p-values for Radio and TV suggest that only they are related to Sales.

# %%
