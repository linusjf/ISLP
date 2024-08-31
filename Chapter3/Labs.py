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
# # Lab: Linear Regression

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
# Import statsmodels objects

# %%
from statsmodels.stats.outliers_influence import variance_inflation_factor as VIF
from statsmodels.stats.anova import anova_lm

# %% [markdown]
# Import ISLP objects

# %%
from ISLP import load_data
from ISLP.models import (ModelSpec as MS, summarize, poly)

# %% [markdown]
# Inspecting objects and namespaces

# %%
dir()

# %%
A = np.array([3,5,11])
dir(A)

# %%
A.sum()

# %% [markdown]
# ## Simple Linear Regression

# %% [markdown]
# We will use the Boston housing dataset which is in the package ISLP

# %%
Boston = load_data("Boston")
Boston.columns

# %%
len(Boston.columns)

# %%
# Boston?

# %% [markdown]
# Use sm.OLS to fit a simple linear regression

# %%
X = pd.DataFrame({"intercept": np.ones(Boston.shape[0]),
                                       "lstat": Boston["lstat"]})
X.head()


# %% [markdown]
# Extract the response and fit the model.

# %%
y = Boston["medv"]
model = sm.OLS(y, X)
results = model.fit()

# %% [markdown]
# Summarize the results using the ISLP method summarize

# %%
summarize(results)

# %% [markdown]
# ## Using Transformations: Fit and Transform

# %%
design = MS(["lstat"])
design = design.fit(Boston)
X = design.transform(Boston)
X.head()

# %%
design = MS(["lstat"])
design = design.fit_transform(Boston)
X.head()

# %% [markdown]
# Full and exhaustive summary of the fit

# %%
results.summary()

# %% [markdown]
# Fitted coefficients can be retrieved as the *params* attribute of results

# %%
results.params

# %% [markdown]
# ### Computing predictions

# %%
design = MS(["lstat"])
new_df = pd.DataFrame({"lstat": [5,10,15]})
print(new_df)
design = design.fit(new_df)
newX = design.transform(new_df)
newX

# %%
new_predictions = results.get_prediction(newX)
new_predictions.predicted_mean

# %% [markdown]
# We can predict confidence intervals for the predicted values.

# %%
new_predictions.conf_int(alpha=0.05)

# %% [markdown]
# We can obtain prediction intervals for the values which are wider than the confidence intervals since they're for a specific instance of lstat by setting obs=True.

# %%
new_predictions.conf_int(obs=True,alpha=0.05)


# %% [markdown]
# Plot medv and lstat using DataFrame.plot.scatter() and add the regression line to the resulting plot.

# %% [markdown]
# Define our abline function

# %%
def abline(ax, b, m, *args, **kwargs):
  "Add a line with slope m and intercept b to ax"
  xlim = ax.get_xlim()
  ylim = [m * xlim[0] + b, m + xlim[1] + b]
  ax.plot(xlim, ylim, *args, **kwargs)


# %%
ax = Boston.plot.scatter("lstat", "medv")
abline(ax, results.params.iloc[0], results.params.iloc[1], "r--", linewidth=3);

# %% [markdown]
# There is some evidence of non-linearity in the relationship b/w lstat and medv.

# %% [markdown]
# Find the fitted values and residuals of the fit as attributes of the results object as *results.fittedvalues* and *results.resid*.
# The get_influence() method computes various influence measures of the regression.
#

# %%
_, ax = subplots(figsize=(8,8))
ax.scatter(results.fittedvalues, results.resid)
ax.set_xlabel("Fitted values")
ax.set_ylabel("Residuals")
ax.axhline(0, c='k', ls='--');

# %% [markdown]
# On the basis of the residual plot, there is some evidence of non-linearity.

# %% [markdown]
# Leverage statistics can be computed for any number of predictors using the hat_matrix_diag attribute of the value returned by the get_influence() method.

# %%
infl = results.get_influence()
_, ax = subplots(figsize=(8,8))
ax.scatter(np.arange(X.shape[0]),infl.hat_matrix_diag)
ax.set_xlabel("Index")
ax.set_ylabel("Leverage")
high_leverage = np.argmax(infl.hat_matrix_diag)
max_leverage = np.max(infl.hat_matrix_diag)
print(high_leverage, max_leverage)
ax.plot(high_leverage, max_leverage, "ro");

# %% [markdown]
# The np.argmax() function returns the index of the highest valued element of an array. Here, we determine which element has the highest leverage.

# %% [markdown]
# ### Multiple linear regression

# %%
Boston.plot.scatter("age", "medv");
X = MS(["lstat","age"]).fit_transform(Boston)
model1 = sm.OLS(y, X)
results1 = model1.fit()
summarize(results1)

# %%
Boston["logage"] = np.log(Boston["age"])
Boston.plot.scatter("logage", "medv");
X = MS(["lstat","logage"]).fit_transform(Boston)
model1 = sm.OLS(y, X)
results1 = model1.fit()
print(summarize(results1))

# %%
Boston["sqrtage"] = np.sqrt(Boston["age"])
Boston.plot.scatter("sqrtage", "medv");
X = MS(["lstat","sqrtage"]).fit_transform(Boston)
model1 = sm.OLS(y, X)
results1 = model1.fit()
summarize(results1)

# %%
Boston = Boston.drop(columns=["logage","sqrtage"])

# %%
terms = Boston.columns.drop("medv")
terms

# %%
X = MS(terms).fit_transform(Boston)
model = sm.OLS(y, X)
results = model.fit()
summarize(results)

# %%
Age has a high p-value. So how about we drop it from the predictors?

# %%
minus_age = Boston.columns.drop(["medv", "age"])
Xma = MS(minus_age).fit_transform(Boston)
model1 = sm.OLS(y, Xma)
summarize(model1.fit())

# %%
np.unique(Boston["indus"])

# %% [raw]
# Similarly, indus has a high p-value. Let's drop it as well.

# %% [markdown]
# minus_age_indus = Boston.columns.drop(["medv", "age", "indus"])
# Xmai = MS(minus_age_indus).fit_transform(Boston)
# model1 = sm.OLS(y, Xmai)
# results1 = model1.fit()
# summarize(results1)

# %%
We can also observe the F-statistic for the regression.

# %%
(results1.fvalue,results1.f_pvalue)

# %%
### Multivariate Goodness of Fit

# %% [markdown]
# We can access the individual components of results by name.

# %%
dir(results1)

# %% [markdown]
# results.rsquared gives us the R<sup>2</sup> and np.sqrt(results.scale) gives us the RSE.

# %%
print("RSE", np.sqrt(results1.scale))

# %%
("R", results1.rsquared)

# %% [markdown]
# Variance Inflation Factors are sometimes useful to assess the collinearity effect in our regression model.

# %%
### Compute VIFs and List Comprehension

# %%
vals = [VIF(X,i) for i in range(1, X.shape[1])]
print(vals)

# %%
vif  = pd.DataFrame({"vif": vals}, index = X.columns[1:])
print(vif)
("VIF Range:", np.min(vif), np.max(vif))

# %% [markdown]
# The VIFs are not very large. If there was some collinearity between predictors, they would tend to blow up.

# %%
### Interaction terms

# %%
X = MS(["lstat", "age", ("lstat", "age")]).fit_transform(Boston)
model2 = sm.OLS(y, X)
results2 = model2.fit()
summarize(results2)

# %% [markdown]
# The interaction terms lstat:age are statistically significant and while the p-value for age exceeds 0.05 ( or 0.01), you do not drop it from the regression since it is a component of a significant interaction.
