# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.4
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Lab - Logistic Regression, LDA, QDA, and KNN

# %% [markdown]
# ## Import notebook functions

# %%
from notebookfuncs import *

# %% [markdown]
# ## Examine the Smarket data &mdash; part of the ISLP library.

# %% [markdown]
# ### Consists of percentage returns for the S&P 500  stock index over 1,250 days, from the beginning of 2001 until the end of 2005.

# %% [markdown]
# For each date, we have recorded the percentage returns for each of the five previous trading days, Lag1 through Lag5. We have also recorded Volume (the number of shares traded on the previous day, in billions), Today (the percentage return on the date in question) and Direction (whether the market was Up or Down on this date).

# %% [markdown]
# ## Import the libraries

# %%
import numpy as np
import pandas as pd
from matplotlib.pyplot import subplots
import statsmodels.api as sm
from ISLP import load_data
from ISLP.models import (ModelSpec as MS , summarize)

# %% [markdown]
# ## New imports for this lab

# %%
from ISLP import confusion_table
from ISLP.models import contrast
from sklearn. discriminant_analysis import (LinearDiscriminantAnalysis as LDA , QuadraticDiscriminantAnalysis as QDA)
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

# %% [markdown]
# ## Load the Smarket data.

# %%
Smarket = load_data('Smarket')
Smarket

# %%
Smarket.columns

# %%
Smarket.Direction = Smarket.Direction.astype("category")

# %%
Smarket.corr(numeric_only=True)

# %% [markdown]
# - As one would expect, the correlations between the lagged return variables and today’s return are close to zero. (Why? [Random walk](https://www.investopedia.com/terms/r/randomwalktheory.asp)?) The only substantial correlation is between Year and Volume. By plotting the data we see that Volume is in creasing over time. In other words, the average number of shares traded daily increased from 2001 to 2005.

# %%
Smarket.plot(y='Volume');

# %% [markdown]
# ## Logistic Regression

# %% [markdown]
# ### Fit a logistic regression model to predict Direction using Lag1 through Lag5 and Volume. 

# %% [markdown]
# We use the sm.GLM() function which fits Generalized Linear Models (GLMs) which includes logistic regression. We could alos sm.Logit() which fits a logit model directly.
# The syntax of sm.GLM() is similar to that of sm.OLS(), except that we must pass in the argument family=sm.families.Binomial() in order to tell statsmodels to run a logistic regression rather than some other type of GLM.

# %%
allvars = Smarket.columns.drop(['Today', 'Direction', 'Year'])
design = MS(allvars)
X = design.fit_transform(Smarket)
y = Smarket.Direction == 'Up'
family = sm.families.Binomial()
glm = sm.GLM(y, X, family=family)
results = glm.fit()
summarize(results)

# %%
(results.pvalues.idxmin(), results.pvalues.min())

# %% [markdown]
# - The smallest p-value here is associated with Lag1. 
# - The negative coefficient for this predictor suggests that if the market had a positive return yesterday then it is less likely to go up today.
# - However, at a value of 0.15, the p-value is still relatively large.
# - So there is no clear evidence of a real association between Lag1 and Direction.

# %%
results.params

# %% [markdown]
# ### Predict 

# %% [markdown]
# The predict() method of results can be used to predict the probability that the market will go up, given values of the predictors. This method returns predictions on the probability scale. If no data set is supplied to the predict() function, then the probabilities are computed for the training data that was used to fit the logistic regression model. As with linear regression, one can pass an optional exog argument consistent with a design matrix if desired. Here we have printed only the first ten probabilities.

# %%
probs = results.predict()
probs[:10]

# %%
allDone();
