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

# %% [markdown]
# ### Imports for python objects and libraries

# %% [markdown]
# #### Set up IPython libraries for customizing notebook display

# %%
from notebookfuncs import *

# %% [markdown]
# #### Import standard libraries

# %%
import numpy as np
import pandas as pd
pd.set_option('display.max_rows', 1000)
pd.set_option('display.max_columns', 1000)
pd.set_option('display.width', 1000)
pd.set_option("display.max.colwidth", None)
import matplotlib.pyplot as plt
import seaborn as sns
import itertools

# %% [markdown]
# #### Statsmodels imports

# %%
import statsmodels.api as sm

# %% [markdown]
# #### Import statsmodels.objects

# %%
from statsmodels.stats.outliers_influence import summary_table


# %% [markdown]
# #### Import ISLP objects

# %%
import ISLP
from ISLP import models
from ISLP import load_data
from ISLP.models import (ModelSpec as MS, summarize, poly)

# %% [markdown]
# #### Import user functions

# %%
from userfuncs import display_residuals_plot
from userfuncs import identify_least_significant_feature
from userfuncs import calculate_VIFs
from userfuncs import identify_highest_VIF_feature
from userfuncs import standardize
from userfuncs import perform_analysis

# %% [markdown]
# #### Set level of significance (alpha)

# %%
LOS_Alpha = 0.01;

# %% [markdown]
# ### Data Cleaning and exploratory data analysis

# %%
Auto = load_data('Auto')
Auto = Auto.sort_values(by=['year'], ascending=True)
Auto.head()
Auto.columns
Auto = Auto.dropna()
Auto.shape
Auto.describe()

# %% [markdown]
# #### Convert origin to categorical type

# %%
Auto["origin"] = Auto["origin"].astype("category")
Auto['origin'] = Auto['origin'].cat.rename_categories({1:'America', 2:'Europe', 3:'Japan'})
Auto.describe()

# %% [markdown]
# ## Create two datasets based on whether the car models have been exposed to the 1973 oil shock or not

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
# #### Standardize numeric variables in the model

# %%
# standardizing dataframes
Auto_preos = Auto_preos.apply(standardize)
Auto_postos = Auto_postos.apply(standardize)
Auto_preos.head()
Auto_postos.head()
Auto_preos.describe()
Auto_postos.describe()

# %% [markdown]
# #### Encode categorical variables as dummy variables dropping the first to remove multicollinearity.

# %%
Auto_preos = pd.get_dummies(Auto_preos, columns=list(["origin"]), drop_first = True, dtype = np.uint8)
Auto_preos.columns

# %%
Auto_postos = pd.get_dummies(Auto_postos, columns=list(["origin"]), drop_first = True, dtype = np.uint8)
Auto_postos.columns

# %% [markdown]
# ### Analysis for pre-oil shock model

# %% [markdown]
# #### Test for multicollinearity using correlation matrix and variance inflation factors

# %%
Auto_preos.corr(numeric_only=True)

# %%
vifdf = calculate_VIFs("mpg ~ " + " + ".join(Auto_preos.columns) + " - mpg", Auto_preos)
vifdf

# %%
identify_highest_VIF_feature(vifdf)

# %%
vifdf = calculate_VIFs("mpg ~ " + " + ".join(Auto_preos.columns) + " - mpg - displacement", Auto_preos)
vifdf

# %%
identify_highest_VIF_feature(vifdf)

# %%
vifdf = calculate_VIFs("mpg ~ " + " + ".join(Auto_preos.columns) + " - mpg - displacement - weight ", Auto_preos)
vifdf

# %%
identify_highest_VIF_feature(vifdf)

# %%
vifdf = calculate_VIFs("mpg ~ " + " + ".join(Auto_preos.columns) + " - mpg - displacement - weight - cylinders ", Auto_preos)
vifdf

# %%
identify_highest_VIF_feature(vifdf)

# %% [markdown]
# #### Linear Regression for mpg ~ horsepower + acceleration + year + origin_Europe + origin_Japan

# %%
cols = list(Auto_preos.columns)
cols.remove("mpg")
cols.remove("displacement")
cols.remove("cylinders")
cols.remove("weight")
formula = ' + '.join(cols)
results = perform_analysis("mpg",formula,Auto_preos);

# %%
identify_least_significant_feature(results, alpha=LOS_Alpha)

# %% [markdown]
# #### Linear Regression after dropping year in pre-oil shock. The model now is mpg ~ horsepower + acceleration + origin_Europe + origin_Japan

# %%
cols.remove("year")
formula = ' + '.join(cols)
results = perform_analysis("mpg",formula,Auto_preos);

# %%
identify_least_significant_feature(results, alpha=LOS_Alpha)

# %% [markdown]
# #### Residual plot for model that drops year for pre-oil shock

# %%
display_residuals_plot(results)

# %% [markdown]
# ### Analysis for post Oil Shock

# %% [markdown]
# #### Linear Regression Analysis for post oil shock using all features

# %%
cols = list(Auto_postos.columns)
cols.remove("mpg")
formula = ' + '.join(cols)
results = perform_analysis("mpg",formula,Auto_postos);

# %% [markdown]
# #### Residual plot for all variables model for post-oil shock

# %%
display_residuals_plot(results)

# %% [markdown]
# ## Finished

# %%
allDone()

# %%
