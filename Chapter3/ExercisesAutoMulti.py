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

# %% [markdown]
# ### Encode categorical variables as dummy variables dropping the first to remove multicollinearity.

# %%
Auto = pd.get_dummies(Auto, columns=list(["cylinders", "origin", "year"]), drop_first = True, dtype = np.uint8)
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
# The coefficients for the year variable range over the following values:
#
# | Year   | Coefficient | 
# | ------- | --------- |
# | year_71	| 0.9104 |	
# | year_72 |	-0.4903	|
# | year_73	| -0.5529	|
# | year_74	| 1.2420	|
# | year_75	| 0.8704	|
# | year_76	| 1.4967	|
# | year_77	| 2.9987	|
# | year_78	| 2.9738	|
# | year_79	| 4.8962	|
# | year_80	| 9.0589	|
# | year_81	| 6.4582	|
# | year_82 |	7.8376	|
#
#
# which suggest that that except for years 72 and 73 (which are not statistically significant), the mpg increases over the base year 1970 by the coefficient value for that year. The most improvement is seen in the year 1980 where the mileage increases by 9 units over the base mpg. 
#
# Note: Year has been converted to a cetegorical variable to better capture the effect of each year. 

# %% [markdown]
# ### (d) Produce some of diagnostic plots of the linear regression fit as described in the lab. Comment on any problems you see with the fit. Do the residual plots suggest any unusually large outliers? Does the leverage plot identify any observations with unusually high leverage?

# %% [markdown]
# ### (e) Fit some models with interactions as described in the lab. Do any interactions appear to be statistically significant?

# %% [markdown]
# ### (f) Try a few  different transformations of the variables, such as log(X), √X, X<sup>2</sup> . Comment on your findings.

# %%
