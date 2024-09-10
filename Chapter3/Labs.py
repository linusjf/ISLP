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

# %%
from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"

## Import up sound alert dependencies
from IPython.display import Audio, display

def allDone():
  url = "https://sound.peal.io/ps/audios/000/064/733/original/youtube_64733.mp3"
  display(Audio(url=url, autoplay=True))



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

# %%
ax = Boston.plot.scatter("lstat", "medv")
ax.axline((ax.get_xlim()[0],results.params.iloc[0]), slope=results.params.iloc[1], color="r", linewidth=3);

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
resultslog = model1.fit()
print(summarize(resultslog))

# %%
Boston["sqrtage"] = np.sqrt(Boston["age"])
Boston.plot.scatter("sqrtage", "medv");
X = MS(["lstat","sqrtage"]).fit_transform(Boston)
model1 = sm.OLS(y, X)
resultssqrt = model1.fit()
summarize(resultssqrt)

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

# %% [markdown]
# We can also observe the F-statistic for the regression.

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

# %%
(results2.rsquared, " > ", results1.rsquared)

# %% [markdown]
# The interaction terms lstat:age are not statistically significant at 0.01 level of significance, and R<sup>2</sup> does not significantly explain the variation in the model. Suffice to say, the interaction term can be dropped.

# %% [markdown]
# ### Non-linear transformation of the predictors

# %% [markdown]
# The poly() function specifies the first argument term to be added to the model matrix

# %%
X = MS([poly("lstat", degree = 2), "age"]).fit_transform(Boston)
model3 = sm.OLS(y, X)
results3 = model3.fit()
summarize(results3)

# %% [markdown]
# The effectively 0 p-value associated with the quadratic term suggests an improved model. The R<sup>2</sup> confirms it

# %%
print(results3.rsquared, " > ", results2.rsquared)

# %% [markdown]
# By default, poly() creates a basis matrix for inclusion in the model matrix whose columns are orthogonal polynomials which are designed for stable least squares computations. If we had included another argument, raw = True , the basis matrix would consist of lstat and lstat ** 2. Both represent quadratic polynomials. The fitted values would not change. Just the polynomial coefficients. The columns created by poly() do not include an intercept column. These are provided by MS().

# %% [markdown]
# Questions:
# What are orthogonal polynomials?
#
# <http://home.iitk.ac.in/~shalab/regression/Chapter12-Regression-PolynomialRegression.pdf>
#
# <https://stats.stackexchange.com/questions/258307/raw-or-orthogonal-polynomial-regression>

# %%
X = MS([poly("lstat", degree = 2, raw = True), "age"]).fit_transform(Boston)
model3 = sm.OLS(y, X)
results3 = model3.fit()
summarize(results3)

# %%
print(results3.rsquared, " > ", results1.rsquared)

# %% [markdown]
# Use the anova_lm() function to further quantify the superiority of the quadratic fit.

# %%
anova_lm(results1, results3)

# %% [markdown]
# results1 corresponds to the linear model containing predictors lstat and age only.
# results3 includes the quadratic term in lstat.
# The anova_lm() function performs a hypothesis test on the two models.
# H<sub>0</sub>: The quadratic term in the model is not needed.
# H<sub>a</sub>: The larger model including the quadratic term is superior.
# Here, the F-statistic is 177.28 and the associated p-value is 0.
# The F-statistic is the t-statistic squared for the quadratic term in results3.
# These nested models differ by 1 degree of freedom.
# This provides very clear evidence that the quadratic term improves the model.
# The anova_lm() function can take more than two models as input.
# The comparison is successive pair-wise.
# That explains the NaNs in the first row of the output above, since there is no previous model with which to compare the output.

# %% [markdown]
# We can further plot the residuals of the regression against the fitted values to check of there still is a pattern discernible.

# %%
_, ax = subplots(figsize=(8,8))
ax.scatter(results3.fittedvalues, results3.resid)
ax.set_xlabel("Fitted values")
ax.set_ylabel("Residuals")
ax.axhline(0, c="k", ls="--");

# %% [markdown]
# We can also try and add the interaction term (lstat, age) to the regression and check the results.

# %%
X = MS([poly("lstat", degree = 2, raw = True), "age", ("lstat","age")]).fit_transform(Boston)
model4 = sm.OLS(y, X)
results4 = model4.fit()
summarize(results4)

# %%
print(results4.rsquared, " > ", results3.rsquared)

# %% [markdown]
# The R<sup>2</sup> in the interaction model again does not exceedingly explain the variance in the model compared to simply having the quadratic term.

# %%
### Qualitative Predictors

# %% [markdown]
# Carseats data

# %%
Carseats = load_data("Carseats")
Carseats.columns

# %%
Carseats.shape

# %%
Carseats.describe()

# %% [markdown]
# ModelSpec() generates dummy variables for categorical columns automatically. This is termed a one-hot encoding of the categorical feature.

# %% [markdown]
# Their columns sum to one. To avoid collinearity with the intercept, the first column is dropped.

# %% [markdown]
# Below we fit a multiple regression model with interaction terms.

# %%
allvars = list(Carseats.columns.drop("Sales"))
y = Carseats["Sales"]
final = allvars + [("Income","Advertising"), ("Price", "Age")]
X = MS(final).fit_transform(Carseats)
model = sm.OLS(y, X)
summarize(model.fit())

# %% [markdown]
# It can be seen that ShelvLoc is significant and a good shelving location is associated with high sales (relative to a bad location). Medium has a smaller coefficient than Good leading us to believe that it leads to higher sales than a bad location, but lesser than a good location.

# %%
allDone()
