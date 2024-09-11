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
# ## Import standard libraries

# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.pyplot import subplots
import seaborn as sns

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

# %% [markdown]
# ### (b) Compute the matrix of correlations between the variables using the DataFrame.corr() method.

# %%
Auto.corr()

# %% [markdown]
# ### (c) Use the sm.OLS() function to perform a multiple linear regression with mpg as the response and all other variables except name as the predictors. Use the summarize() function to print the results. Comment on the output. For instance:

# %% [markdown]
# ## Convert cylinders, year and origin columns to categorical types

# %%
Auto["cylinders"] = Auto["cylinders"].astype("category")
Auto["origin"] = Auto["origin"].astype("category")
Auto["year"] = Auto["year"].astype("category")
Auto.describe()

# %%
sns.relplot(Auto, x="year", y="weight", col="origin", hue="cylinders", style="cylinders", estimator='mean', kind="line");

# %% [markdown]
# #### The weight of the 8-cylinder American made models show a decline from the highs of 1972. It can also be seen that American made cars are heavier than their European and Japanese counterparts especially in the most common models with 4 cylinders.

# %%
sns.relplot(Auto, x="year", y="mpg", col="origin", hue="cylinders", style="cylinders", estimator='mean', kind="line");


# %% [markdown]
# #### It can be seen that after the [oil shock of 1973](https://en.wikipedia.org/wiki/1973_oil_crisis) and the regulations and actions taken by the US government, the mileage for American made cars rose across all models. This was, however, matched by the European and Japanese models which were already lighter and more fuel efficient.

# %% [markdown]
# ### Encode categorical variables as dummy variables dropping the first to remove multicollinearity.

# %%
def categorize_for_oil_shock(row):
  # we add 3 years because it takes approximately that long for car manufacturers to introduce a new model
  if row["year"] in (70, 71, 72, 73, 74, 75, 76):
    return 0;
  return 1;

Auto["oilshock"] = Auto.apply(categorize_for_oil_shock, axis=1);

# %%
Auto.boxplot(column="mpg", by=["oilshock", "origin"]);

# %%
Auto_os = Auto.drop(["year"], axis = 1)
Auto_os.columns
Auto_os = pd.get_dummies(Auto_os, columns=list(["cylinders", "origin"]), drop_first = True, dtype = np.uint8)
Auto_os.columns

# %%
y = Auto_os["mpg"]
Auto_os.columns.drop("mpg")

# %%
cols = list(Auto_os.columns)
cols.remove("mpg")
X = MS(cols).fit_transform(Auto_os)
formula = ' + '.join(cols)
model = smf.ols(f'mpg ~ {formula}', data=Auto_os)
results = model.fit()
results.summary()

# %% [markdown]
# #### i. Is there a relationship between the predictors and the response? Use the anova_lm() function from statsmodels to answer this question.
# #### ii. Which predictors appear to have a statistically significant relationship to the response? 
# #### iii. What does the coefficient for the year variable suggest?

# %%
anova_lm(results)

# %% [markdown]
# #### There seems to be a statistical relationship between all of the predictors and the response variable, mpg, except for acceleration. 
# #### Even though some of the categorical variables are insignificant, even if one of the levels is significant, it is advisable to retain them all in the model. 
#
# <https://stats.stackexchange.com/questions/24298/can-i-ignore-coefficients-for-non-significant-levels-of-factors-in-a-linear-mode>
#
# Note: Year has been converted to a categorical variable oilshock to better capture the effects of the oil shock of 1973 on the mileage.

# %% [markdown]
# ### (d) Produce some of diagnostic plots of the linear regression fit as described in the lab. Comment on any problems you see with the fit. Do the residual plots suggest any unusually large outliers? Does the leverage plot identify any observations with unusually high leverage?

# %% [markdown]
# #### Before producing the diagnostic plots, let's first create the model suggested hy the analysis above by dropping the feature acceleration from the model.

# %%
cols.remove("acceleration")
X = MS(cols).fit_transform(Auto_os)
formula = ' + '.join(cols)
model = smf.ols(f'mpg ~ {formula}', data=Auto_os)
results = model.fit()
results.summary()

# %% [markdown]
# #### From the modified model, it is evident that displacement is not statistically significant. This is, perhaps, because of the collinearity with number of cylinders. So we can drop it from the model as well.

# %%
cols.remove("displacement")
X = MS(cols).fit_transform(Auto_os)
formula = ' + '.join(cols)
model = smf.ols(f'mpg ~ {formula}', data=Auto_os)
results = model.fit()
results.summary()

# %%
display("Thus, the final model is the one below: ")
display("mpg ~ " + formula)


# %% [markdown]
# #### We can now try and plot the diagnostics for the model.

# %%
TSS = np.sum((y - np.mean(y)) ** 2)
TSS
RSS = np.sum((y - results.fittedvalues) ** 2)
RSS
RSE = np.sqrt(RSS/results.df_model)
RSE

# %% [markdown]
# #### Auto data results
# | Quantity | Value |
# | -------- | ------ |
# | RSE | 21.08 |
# | R<sup>2</sup> Adjusted | 0.828 |
#

# %% [markdown]
# ### (e) Fit some models with interactions as described in the lab. Do any interactions appear to be statistically significant?

# %% [markdown]
# ### (f) Try a few  different transformations of the variables, such as log(X), √X, X<sup>2</sup> . Comment on your findings.

# %%
allDone()

# %%

# %%
