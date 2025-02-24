# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
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

# %% [raw]
# {{< include codewraplatex.yml >}}

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
from sklearn.discriminant_analysis import (LinearDiscriminantAnalysis as LDA , QuadraticDiscriminantAnalysis as QDA)
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

# %% [markdown]
#  We begin by performing LDA on the Smarket data, using the function LinearDiscriminantAnalysis(), which we have abbreviated LDA(). We fit the model using only the observations before 2005.

# %%
lda = LDA( store_covariance=True)

# %% [markdown]
# Since the LDA estimator automatically adds an intercept, we should remove the column corresponding to the intercept in both X_train and X_test. We can also directly use the labels rather than the Boolean vectors y_train.

# %%
X_train , X_test = [M.drop(columns =['intercept']) for M in [X_train , X_test ]]
lda.fit(X_train , L_train)

# %% [markdown]
# Having fit the model, we can extract the means in the two classes with the means_ attribute. These are the average of each predictor within each class, and are used by LDA as estimates of µk . These suggest that there is a tendency for the previous 2 days’ returns to be negative on days when the market increases, and a tendency for the previous days’ returns to be positive on days when the market declines.

# %%
lda.means_

# %% [markdown]
# The estimated prior probabilities are stored in the priors_ attribute. The package sklearn typically uses this trailing _ to denote a quantity estimated when using the fit() method. We can be sure of which entry corresponds to which label by looking at the classes_ attribute.

# %%
lda.classes_

# %% [markdown]
# The LDA output indicates that $\hat{\pi}_{Down}$ = 0.492 and $\hat{\pi}_{Up}$ = 0.508.

# %%
lda.priors_

# %% [markdown]
# The linear discriminant vectors can be found in the scalings_ attribute:

# %%
lda.scalings_

# %% [markdown]
# These values provide the linear combination of Lag1 and Lag2 that are used to form the LDA decision rule. In other words, these are the multipliers of the elements of X = x in (4.24). 
# $$
# {\large \delta_k =  x^T\Sigma^{-1}\mu_k + \frac {\mu_k^T\Sigma^{-1}\mu_k} {2} + log(\pi_k) }
# $$
# If −0.64 × Lag1 − 0.51 × Lag2 is large, then the LDA classifier will predict a market increase, and if it is small, then the LDA classifier will predict a market decline.

# %%
lda_pred = lda.predict(X_test)

# %% [markdown]
# As we observed in our comparison of classification methods (Section 4.5), the LDA and logistic regression predictions are almost identical.

# %%
confusion_table(lda_pred , L_test)

# %% [markdown]
# We can also estimate the probability of each class for each point in a training set. Applying a 50% threshold to the posterior probabilities of being in class one allows us to recreate the predictions contained in lda_pred.

# %%
lda_prob = lda.predict_proba(X_test)
np.all(np.where(lda_prob [:,1] >= 0.5, 'Up','Down') == lda_pred)

# %% [markdown]
# Above, we used the np.where() function that creates an array with value 'Up' for indices where the second column of lda_prob (the estimated posterior probability of 'Up') is greater than 0.5. For problems with more than two classes the labels are chosen as the class whose posterior probability is highest:

# %%
np.all( [lda.classes_[i] for i in np.argmax(lda_prob , 1)] == lda_pred )

# %% [markdown]
# If we wanted to use a posterior probability threshold other than 50% in order to make predictions, then we could easily do so. For instance, suppose that we wish to predict a market decrease only if we are very certain that the market will indeed decrease on that day — say, if the posterior probability is at least 90%. We know that the first column of lda_prob corresponds to the label Down after having checked the classes_ attribute, hence we use the column index 0 rather than 1 as we did above.

# %%
np.sum(lda_prob [:,0] > 0.9)

# %% [markdown]
# No days in 2005 meet that threshold! In fact, the greatest posterior probability of decrease in all of 2005 was 52.02%.

# %% [markdown]
# ## Quadratic Discriminant Analysis

# %% [markdown]
# ### Fit a QDA model to the Smarket data.

# %%
qda = QDA(store_covariance=True)
qda.fit(X_train , L_train)

# %% [markdown]
# The QDA() function will again compute means_ and priors_.

# %%
qda.means_ , qda.priors_

# %% [markdown]
# The QDA() classifier will estimate one covariance per class. Here is the estimated covariance in the first class:

# %%
qda.covariance_[0]

# %% [markdown]
# List all the covariances.

# %%
qda.covariance_

# %% [markdown]
# The output contains the group means. But it does not contain the coefficients of the linear discriminants, because the QDA classifier involves a quadratic, rather than a linear function. The predict() function works in exactly the same fashion as for LDA.

# %%
qda_pred = qda.predict(X_test)
confusion_table(qda_pred, L_test)

# %% [markdown]
# Interestingly, the QDA predictions are accurate almost 60% of the time, even though the 2005 data was not used to fit the model.

# %%
np.mean(qda_pred == L_test)

# %% [markdown]
# This level of accuracy is quite impressive for stock market data, which is known to be quite hard to model accurately. This suggests that the quadratic form assumed by QDA may capture the true relationship more accurately than the linear forms assumed by LDA and logistic regression. However, we recommend evaluating this method’s performance on a larger test set before betting that this approach will consistently beat the market!

# %%
The score() function is an alternate way of obtaining the level of accuracy.

# %%
qda.score(X_test, L_test)

# %%
allDone();
