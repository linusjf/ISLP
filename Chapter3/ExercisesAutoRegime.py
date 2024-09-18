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

# %% [markdown] jp-MarkdownHeadingCollapsed=true
# #### Set up IPython libraries for customizing notebook display

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
# #### Import standard libraries

# %%
import numpy as np
from scipy import stats
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
# #### Statsmodels imports

# %%
import statsmodels.api as sm

# %% [markdown]
# #### Import statsmodels.objects

# %%
from statsmodels.stats.outliers_influence import variance_inflation_factor as VIF
from statsmodels.stats.outliers_influence import summary_table
from statsmodels.stats.anova import anova_lm
import statsmodels.formula.api as smf
from patsy import dmatrices

# %% [markdown]
# #### Import ISLP objects

# %%
import ISLP
from ISLP import models
from ISLP import load_data
from ISLP.models import (ModelSpec as MS, summarize, poly)


# %% [markdown]
# #### Define user functions

# %% [markdown]
# ##### Display residuals plot function

# %%
def display_residuals_plot(results):
  _, ax = subplots(figsize=(8,8))
  ax.scatter(results.fittedvalues, results.resid)
  ax.set_xlabel("Fitted values for " + results.model.endog_names)
  ax.set_ylabel("Residuals")
  ax.axhline(0, c="k", ls="--");


# %% [markdown]
# ##### Identify least statistically significant variable or column

# %%
def identify_least_significant_feature(results, alpha=0.05):
  highest_pvalue = results.pvalues.iloc[np.argmax(results.pvalues)]
  if highest_pvalue > alpha:
    variable = results.pvalues.index[np.argmax(results.pvalues)]
    display("We find the least significant variable in this model is " + variable + " with a p-value of " + str(highest_pvalue))
    display("Using the backward methodology, we drop " + variable + " from the new model")
  else:
    display("No variables are statistically insignificant.")
    display("The model " + results.model.formula + " cannot be pruned further.")


# %% [markdown]
# ##### Calculate [Variance Inflation Factors(VIFs) for features in a model](https://www.statology.org/how-to-calculate-vif-in-python/)

# %%
def calculate_VIFs(formula, df):
  # find design matrix for linear regression model using formula and dataframe
  _, X = dmatrices(formula, data=df, return_type='dataframe')
  # calculate VIF for each explanatory variable
  vif = pd.DataFrame()
  vif['VIF'] = [VIF(X.values, i) for i in range(1, X.shape[1])]
  vif['Feature'] = X.columns[1:]
  vif = vif.set_index(["Feature"])
  return vif


# %% [markdown]
# ##### Identify feature with highest VIF

# %%
def identify_highest_VIF_feature(vifdf, threshold=5):
  highest_vif = vifdf["VIF"].iloc[np.argmax(vifdf)]
  if highest_vif > threshold:
    variable = vifdf.index[np.argmax(vifdf["VIF"])]
    display("We find the highest VIF in this model is " + variable + " with a VIF of " + str(highest_vif))
    display("Hence, we drop " + variable + " from the model to be fitted.")
  else:
    display("No variables are significantly collinear.")


# %% [markdown]
# #### Set level of significance (alpha)

# %%
LOS_Alpha = 0.01

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

# %% [markdown] jp-MarkdownHeadingCollapsed=true
# #### Convert year and origin columns to categorical types

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
# #### Encode categorical variables as dummy variables dropping the first to remove multicollinearity.

# %%
Auto_preos = pd.get_dummies(Auto_preos, columns=list(["origin"]), drop_first = True, dtype = np.uint8)
Auto_preos.columns
Auto_postos = pd.get_dummies(Auto_postos, columns=list(["origin"]), drop_first = True, dtype = np.uint8)
Auto_postos.columns

# %% [markdown]
# ### Analysis for pre-oil shock model

# %% [markdown]
# #### Standardize numeric variables in the model

# %%
# standardizing dataframe
Auto_preos["mpg"] = (Auto_preos["mpg"] - Auto_preos["mpg"].mean())/Auto_preos["mpg"].std()
Auto_preos["cylinders"] = (Auto_preos["cylinders"] - Auto_preos["cylinders"].mean())/Auto_preos["cylinders"].std()
Auto_preos["horsepower"] = (Auto_preos["horsepower"] - Auto_preos["horsepower"].mean())/Auto_preos["horsepower"].std()
Auto_preos["weight"] = (Auto_preos["weight"] - Auto_preos["weight"].mean())/Auto_preos["weight"].std()
Auto_preos["acceleration"] = (Auto_preos["acceleration"] - Auto_preos["acceleration"].mean())/Auto_preos["acceleration"].std()
Auto_preos["year"] = (Auto_preos["year"] - Auto_preos["year"].mean())/Auto_preos["year"].std()
Auto_preos["displacement"] = (Auto_preos["displacement"] - Auto_preos["displacement"].mean())/Auto_preos["displacement"].std()
Auto_preos.describe()

# %% [markdown]
# #### Test for multicollinearity using correlation matrix and variance inflation factors

# %%
Auto_preos.corr()

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
model = smf.ols(f'mpg ~ {formula}', data=Auto_preos)
results = model.fit()
results.summary()
anova_lm(results)

# %% [markdown]
# #### Residual plot for all variables model for pre-oil shock

# %%
display_residuals_plot(results)

# %%
identify_least_significant_feature(results, alpha=LOS_Alpha)

# %% [markdown]
# We do not wish to drop the Intercept. The next least significant variable is year. We drop this from the model.

# %% [markdown]
# #### Linear Regression after dropping year in pre-oil shock. The model now is mpg ~ horsepower + acceleration + origin_Europe + origin_Japan

# %%
cols.remove("year")
formula = ' + '.join(cols)
model = smf.ols(f'mpg ~ {formula}', data=Auto_preos)
results = model.fit()
results.summary()
anova_lm(results)

# %% [markdown]
# #### Residual plot for model that drops year for pre-oil shock

# %%
display_residuals_plot(results)

# %%
identify_least_significant_feature(results, alpha=LOS_Alpha)

# %% [markdown]
# ### Analysis for post Oil Shock

# %% [markdown]
# #### Linear Regression Analysis for post oil shock using all features

# %%
cols = list(Auto_postos.columns)
cols.remove("mpg")
formula = ' + '.join(cols)
model = smf.ols(f'mpg ~ {formula}', data=Auto_postos)
results = model.fit()
results.summary()
anova_lm(results)

# %% [markdown]
# #### Residual plot for all variables model for post-oil shock

# %%
display_residuals_plot(results)

# %% [markdown]
# ## Finished

# %%
allDone()

# %%
