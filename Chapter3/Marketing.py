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
from statsmodels.stats.outliers_influence import summary_table
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

# %% [markdown]
# ## How large is the association between each medium and sales?

# %%
results.conf_int(alpha=0.05)

# %% [markdown]
# The confidence intervals for TV and Radio are narrow and far from zero. This provides evidence that these media are related to sales.
#
#

# %% [markdown]
# The interval for Newspaper includes zero indicating that it is not statistically significant given values of TV and Radio.

# %%
vals = [VIF(X, i) for i in range(1, X.shape[1])]
print(vals)

# %% [markdown]
# The VIF scores are 1.005, 1.145 and 1.145 respectively for TV, radio and newspaper. These suggest no evidence of collinearity as an explnation for wide standard errors for newspaper.

# %% [markdown]
# In order to assess the association of each medium individually on sales, we can perform three separate linear regressions.

# %%
TV = MS(["TV"]).fit_transform(Advertising)
model = sm.OLS(y, TV)
results = model.fit()
print(summarize(results))
Radio = MS(["Radio"]).fit_transform(Advertising)
model = sm.OLS(y, Radio)
results = model.fit()
print(summarize(results))
Newspaper = MS(["Newspaper"]).fit_transform(Advertising)
model = sm.OLS(y, Newspaper)
results = model.fit()
print(summarize(results))

# %% [markdown]
# Looking at the p-values, there is evidence of a strong association b/w TV and sales and Radio and sales. There is evidence of a mild association between Newspaper and sales when TV and radio are ignored.

# %%
## How accurately can we predict future sales?

# %%
## Is the relationship linear?

# %%
## Is there synergy among the advertising media?
