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
Smarket.describe()

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

# %% [markdown]
# In order to make a prediction as to whether the market will go up or down on a particular day, we must convert these predicted probabilities into class labels, Up or Down.

# %%
labels = np.array(['Down']*1250)
labels[probs >0.5] = "Up"

# %% [markdown]
# The confusion_table() function from the ISLP package summarizes these confusion predictions, showing how many observations were correctly or incorrectly classified. Our function, which is adapted from a similar function in the module sklearn.metrics, transposes the resulting matrix and includes row and column labels. The confusion_table() function takes as first argument the predicted labels, and second argument the true labels.

# %%
confusion_table(labels , Smarket.Direction)

# %% [markdown]
# The diagonal elements of the confusion matrix indicate correct predictions, while the off-diagonals represent incorrect predictions. Hence our model correctly predicted that the market would go up on 507 days and that it would go down on 145 days, for a total of 507 + 145 = 652 correct predictions. The np.mean() function can be used to compute the fraction of days for which the prediction was correct. In this case, logistic regression correctly predicted the movement of the market 52.2% of the time.

# %%
(507+145) /1250 , np.mean(labels == Smarket.Direction)

# %% [markdown]
# At first glance, it appears that the logistic regression model is working a little better than random guessing. However, this result is misleading because we trained and tested the model on the same set of 1,250 observations. In other words, 100 − 52.2 = 47.8% is the training error rate. As we have seen previously, the training error rate is often overly optimistic — it tends to underestimate the test error rate. In order to better assess the accuracy of the logistic regression model in this setting, we can fit the model using part of the data, and then examine how well it predicts the held out data. This will yield a more realistic error rate, in the sense that in practice we will be interested in our model’s performance not on the data that we used to fit the model, but rather on days in the future for which the market’s movements are unknown.

# %% [markdown]
# ### Train and Test

# %% [markdown]
# To implement this strategy, we first create a Boolean vector corresponding to the observations from 2001 through 2004. We then use this vector to create a held out data set of observations from 2005.

# %%
train = (Smarket.Year < 2005)
Smarket_train = Smarket.loc[train]
Smarket_test = Smarket.loc[~train];

# %%
Smarket_train.shape

# %%
Smarket_test.shape

# %% [markdown]
# #### Fit the training data

# %%
X_train , X_test = X.loc[train], X.loc[~train]
y_train , y_test = y.loc[train], y.loc[~train]
glm_train = sm.GLM(y_train , X_train, family=sm.families.Binomial())
results = glm_train.fit()
probs = results.predict(exog=X_test)

# %% [markdown]
# We compare the predictions for 2005 to the actual movements of the market over that time period. We will first store the test and training labels (recall y_test is binary).

# %%
D = Smarket.Direction
L_train , L_test = D.loc[train], D.loc[~train]

# %% [markdown]
# Now we threshold the fitted probability at 50% to form our predicted labels.

# %%
labels = np.array(['Down']*len(L_test))
labels[probs > 0.5] = 'Up'
confusion_table(labels , L_test)

# %% [markdown]
# The test accuracy is about 48% while the error rate is about 52%

# %%
np.mean(labels == L_test), np.mean(labels != L_test)

# %% [markdown]
# The results are rather disappointing: the test error rate is 52%, which is worse than random guessing! One would not generally expect to be able to use previous days’ returns to predict future market performance.

# %% [markdown]
# ### Trying a more effective model

# %% [markdown]
# The p-values in our original regression were quite underwhelming since none of them were less than 0.05, our preferred level of significance.
# Since Lag1 and Lag2 have the lowest p-values, let's drop all the other predictors from our logistic model and check our results.

# %%
model = MS(['Lag1', 'Lag2']).fit(Smarket)
X = model.transform(Smarket)
X_train , X_test = X.loc[train], X.loc[~train]
glm_train = sm.GLM(y_train , X_train, family=sm.families.Binomial())
results = glm_train.fit()
probs = results.predict(exog=X_test)
labels = np.array (['Down']*len(X_test))
labels[probs >0.5] = 'Up'
confusion_table(labels , L_test)

# %% [markdown]
# Let’s evaluate the overall accuracy as well as the accuracy within the days when logistic regression predicts an increase.

# %%
(35+106) /252 ,106/(106+76)

# %% [markdown]
# Now the results appear to be a little better: 56% of the daily movements have been correctly predicted. It is worth noting that in this case, a much simpler strategy of predicting that the market will increase every day will also be correct 56% of the time! Hence, in terms of overall error rate, the logistic regression method is no better than the naive approach. However, the confusion matrix shows that on days when logistic regression predicts an increase in the market, it has a 58% accuracy rate. This suggests a possible trading strategy of buying on days when the model predicts an increasing market, and avoiding trades on days when a decrease is predicted. Of course one would need to investigate more carefully whether this small improvement was real or just due to random chance.

# %% [markdown]
# Suppose that we want to predict the returns associated with particular values of Lag1 and Lag2. In particular, we want to predict Direction on a day when Lag1 and Lag2 equal 1.2 and 1.1, respectively, and on a day when they equal 1.5 and −0.8. We do this using the predict() function.

# %%
newdata = pd.DataFrame ({'Lag1':[1.2 , 1.5], 'Lag2':[1.1 , -0.8]});
newX = model.transform(newdata)
results.predict(newX)

# %% [markdown]
# ## Linear Discriminant Analysis

# %%
allDone();
