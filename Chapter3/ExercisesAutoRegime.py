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
# # Auto dataset two regimes: Pre-oilshock and Post-oilshock

# %% [markdown]
# ## We can also test if there are two regimes that contribute to the heteroskedasticity by running separate regressions for pre-oilshock and post-oilshock.

# %% [markdown] jp-MarkdownHeadingCollapsed=true
# ### Set up IPython libraries for customizing notebook display

# %%
from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"

# %%
## Import up sound alert dependencies
from IPython.display import Audio, display

def allDone():
  url = "https://sound.peal.io/ps/audios/000/064/733/original/youtube_64733.mp3"
  display(Audio(url=url, autoplay=True))


# %% [markdown]
# ### Import standard libraries

# %%
import numpy as np
import pandas as pd
pd.set_option('display.max_rows', 1000)
pd.set_option('display.max_columns', 1000)
pd.set_option('display.width', 1000)
pd.set_option("display.max.colwidth", None)
import matplotlib.pyplot as plt
from matplotlib.pyplot import subplots
import seaborn as sns
import itertools

# %% [markdown]
# ### Statsmodels imports

# %%
import statsmodels.api as sm

# %% [markdown] jp-MarkdownHeadingCollapsed=true
# ### Import statsmodels.objects

# %%
from statsmodels.stats.outliers_influence import variance_inflation_factor as VIF
from statsmodels.stats.outliers_influence import summary_table
from statsmodels.stats.anova import anova_lm
import statsmodels.formula.api as smf

# %% [markdown] jp-MarkdownHeadingCollapsed=true
# ### Import ISLP objects

# %%
import ISLP
from ISLP import models
from ISLP import load_data
from ISLP.models import (ModelSpec as MS, summarize, poly)

# %%
Auto = load_data('Auto')
Auto = Auto.sort_values(by=['year'], ascending=True)
Auto.head()
Auto.columns
Auto = Auto.dropna()
Auto.shape
Auto.describe()

# %% [markdown] jp-MarkdownHeadingCollapsed=true
# ### Convert year and origin columns to categorical types

# %%
Auto["origin"] = Auto["origin"].astype("category")
Auto['origin'] = Auto['origin'].cat.rename_categories({1:'America', 2:'Europe', 3:'Japan'})
Auto.describe()

# %%
## Create two datasets based on whether the car models have been exposed to the 1973 oil shock or not

# %%
Auto_preos = Auto[Auto["year"] <= 76]
Auto_preos.shape
Auto_preos.describe()
Auto_preos.corr(numeric_only=True)

# %%
Auto_postos = Auto[Auto["year"] > 76]
Auto_postos.shape
Auto_postos.describe()

# %%
display("If you look at the two datasets as displayed above, it's evident that the oil shock had a major impact on the models produced since.")
display(Auto_preos.mean(numeric_only=True), Auto_postos.mean(numeric_only=True))
display("Mileage increased, number of cylinders decreased, displacement decreased, horsepower decreased, weight decreased and time to acceleration increased thus indicating that less powerful and less performant cars were produced in the immediate period after the oil shock of 1973.")

# %% [markdown]
# ### Encode categorical variables as dummy variables dropping the first to remove multicollinearity.

# %%
Auto_preos = pd.get_dummies(Auto_preos, columns=list(["origin"]), drop_first = True, dtype = np.uint8)
#Auto_preos = Auto_preos.drop(columns=["cylinders","displacement","acceleration"])
Auto_preos.columns
Auto_postos = pd.get_dummies(Auto_postos, columns=list(["origin"]), drop_first = True, dtype = np.uint8)
#Auto_postos = Auto_postos.drop(columns=["cylinders","displacement","acceleration"])
Auto_postos.columns

# %% [markdown]
# ### Run linear regression for pre Oil Shock

# %%
cols = list(Auto_preos.columns)
cols.remove("mpg")
formula = ' + '.join(cols)
model = smf.ols(f'mpg ~ {formula}', data=Auto_preos)
results = model.fit()
results.summary()
anova_lm(results)

# %% [markdown]
# ### Residual plot for pre oil shock

# %%
_, ax = subplots(figsize=(8,8))
ax.scatter(results.fittedvalues, results.resid)
ax.set_xlabel("Fitted values for mpg")
ax.set_ylabel("Residuals")
ax.axhline(0, c="k", ls="--");

# %%
cols = list(Auto_postos.columns)
cols.remove("mpg")
formula = ' + '.join(cols)
model = smf.ols(f'mpg ~ {formula}', data=Auto_postos)
results = model.fit()
results.summary()
anova_lm(results)

# %% [markdown]
# ### Run linear regression for post Oil Shock

# %% [markdown]
# ### Residual plot for post oil shock

# %%
_, ax = subplots(figsize=(8,8))
ax.scatter(results.fittedvalues, results.resid)
ax.set_xlabel("Fitted values for mpg")
ax.set_ylabel("Residuals")
ax.axhline(0, c="k", ls="--");

# %% [markdown]
# ## Finished

# %%
allDone()

# %%
