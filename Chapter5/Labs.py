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
# # Lab: Cross-Validation and the Bootstrap

# %% [markdown]
# ## Import notebook functions

# %%
from notebookfuncs import *

# %% [markdown]
# ## Import libraries

# %%
import numpy as np
import statsmodels.api as sm
from ISLP import load_data
from ISLP.models import (ModelSpec as MS , summarize ,poly)
from sklearn.model_selection import train_test_split

# %% [markdown]
# ## New libraries needed for Cross-Validation and the Bootstrap

# %%
from functools import partial
from sklearn.model_selection import (cross_validate, KFold , ShuffleSplit)
from sklearn.base import clone
from ISLP.models import sklearn_sm

# %% [markdown]
# ## Validation Set Approach

# %% [markdown]
# We explore the use of the validation set approach in order to estimate the test error rates that result from fitting various linear models on the **Auto** data set.
#
# We use the function `train_test_split()` to split the data into training and validation sets. As there are 392 observations, we split into two equal sets of size 196 using the argument `test_size=196`. It is generally a good idea to set a random seed when performing operations like this that contain an element of randomness, so that the results obtained can be reproduced precisely at a later time. We set the random seed of the splitter with the argument `random_state=0`.

# %%
Auto = load_data('Auto')
Auto_train , Auto_valid = train_test_split(Auto , test_size =196, random_state =0)

# %% [markdown]
# Now we can fit a linear regression using only the observations corresponding to the training set `Auto_train`.

# %%
hp_mm = MS(['horsepower'])
X_train = hp_mm.fit_transform(Auto_train)
y_train = Auto_train['mpg']
model = sm.OLS(y_train , X_train)
results = model.fit();


# %% [markdown]
# We now use the `predict()` method of results evaluated on the model matrix for this model created using the validation data set. We also calculate the validation Mean Square Error (MSE) of our model.

# %%
X_valid = hp_mm.transform(Auto_valid)
y_valid = Auto_valid['mpg']
valid_pred = results.predict(X_valid)
mse = np.mean((y_valid - valid_pred)**2)

# %%
printmd(f"Hence our estimate for the validation MSE of the linear regression fit is **{mse:0.2f}**.")


# %% [markdown]
# We can also estimate the validation error for higher-degree polynomial regressions. We first provide a function `evalMSE()` that takes a model string as well as a training and test set and returns the MSE on the test set.

# %%
def evalMSE(terms,response,train,test):

  mm = MS(terms)
  X_train = mm.fit_transform(train)
  y_train = train[response]

  X_test = mm.transform(test)
  y_test = test[response]

  results = sm.OLS(y_train , X_train).fit()
  test_pred = results.predict(X_test)

  return np.mean((y_test - test_pred)**2)


# %% [markdown]
# Let's use this function to estimate the validation MSE using linear, quadratic and cubic fits. We use the `enumerate()` function here, which gives both the values and indices of objects as one iterates over a for loop.

# %%
MSE = np.zeros(3)
for idx, degree in enumerate(range(1, 4)):
  MSE[idx] = evalMSE([poly('horsepower', degree)], 'mpg', Auto_train, Auto_valid)
MSE

# %%
printmd(f"These error rates are **{MSE[0]:0.2f}**, **{MSE[1]:0.2f}**, and **{MSE[2]:0.2f}**, respectively.")

# %% [markdown]
# If we choose a different training/validation split instead, then we can expect somewhat different errors on the validation set.

# %%
Auto_train , Auto_valid = train_test_split(Auto , test_size = 196, random_state = 3)
MSE = np.zeros(3)
for idx, degree in enumerate(range(1, 4)):
  MSE[idx] = evalMSE([poly('horsepower', degree)], 'mpg', Auto_train, Auto_valid)
MSE

# %%
printmd(f"The error rates now are **{MSE[0]:0.2f}**, **{MSE[1]:0.2f}**, and **{MSE[2]:0.2f}**, respectively.")

# %% [markdown]
# These results are consistent with our previous findings: a model that predicts **mpg** using a quadratic function of **horsepower** performs better than a model that involves only a linear function of **horsepower**, and there is no evidence of an improvement in using a cubic function of **horsepower**.

# %% [markdown]
# ## Cross-Validation

# %% [markdown]
# The simplest way to cross-validate in Python is to use `sklearn`, which has a different interface or API than`statsmodels`.
#
# The `ISLP` package provides a wrapper, `sklearn_sm()`, that enables us to easily use the cross-validation tools of `sklearn` with models fit by `statsmodels
#
# The class `sklearn_sm()` has as its first argument a model from `statsmodels`. It can take two additional optional arguments: `model_str` which can be used to specify a formula, and `model_args` which should be a dictionary of additional arguments used when fitting the model. For example, to fit a logistic regression model we have to specify a family argument. This is passed as `model_args {'family':sm.families.Binomial()}`.

# %%
hp_model = sklearn_sm(sm.OLS, MS(['horsepower']))
X, Y = Auto.drop(columns =['mpg']), Auto['mpg']
cv_results = cross_validate(hp_model, X, Y, cv=Auto.shape[0])
cv_err = np.mean(cv_results['test_score'])
cv_err

# %%
printmd(f"The arguments to `cross_validate()` are as follows: an object with the appropriate `fit()`, `predict()`, and `score()` methods, an array of features X and a response Y. We also included an additional argument `cv` to `cross_validate()`; specifying an integer K results in K-fold cross-validation. We have provided a value corresponding to the total number of observations, which results in leave-one-out cross-validation (LOOCV). The `cross-validate()` function produces a dictionary with several components; we simply want the cross-validated test score here (MSE), which is estimated to be {cv_err:.2f}.");

# %% [markdown]
# We can repeat this procedure for increasingly complex polynomial fits. To automate the process, we again use a for loop which iteratively fits polynomial regressions of degree 1 to 5, computes the associated crossvalidation error, and stores it in the $i_{th}$ vector `cv_error`. The variable `d` in the for loop corresponds to the degree of the polynomial. We begin by initializing the vector. This command may take a couple of seconds to run.

# %%
cv_error = np.zeros (5)
H = np.array(Auto['horsepower'])
M = sklearn_sm(sm.OLS)
for i, d in enumerate(range (1,6)):
  # this sets up the polynomial features for horsepower
  # such as 1, hp, hp**2, hp**3 till hp**5 in the final loop
  X = np.power.outer(H, np.arange(d+1))
  M_CV = cross_validate(M, X, Y, cv=Auto.shape[0])
  cv_error[i] = np.mean(M_CV['test_score'])
cv_error

# %% [markdown]
# We see a sharp drop in the estimated test MSE between the linear and quadratic fits, but then no clear improvement from using higher-degree polynomials.
#
# Above we introduced the `outer()` method of the `np.power()` function. The `outer()` method is applied to an operation that has two arguments, such as `add()`, `min()`, or `power()`. It has two arrays as arguments, and then forms a larger array where the operation is applied to each pair of elements of the two arrays.

# %%
A = np.array ([3, 5, 9])
B = np.array ([2, 4])
np.add.outer(A, B)

# %% [markdown]
# In the CV example above, we used K = n, but of course we can also use K < n. The code is very similar to the above (and is significantly faster). Here we use `KFold()` to partition the data into K = 10 random groups. We use `random_state` to set a random seed and initialize a vector `cv_error` in which we will store the CV errors corresponding to the polynomial fits of degrees one to five.

# %%
cv_error = np.zeros(5)
# use same splits for each degree
cv = KFold(n_splits=10, shuffle=True, random_state=0)
for i, d in enumerate(range (1,6)):
  X = np.power.outer(H, np.arange(d+1))
  M_CV = cross_validate(M, X, Y, cv=cv)
  cv_error[i] = np.mean(M_CV['test_score'])
cv_error

# %% [markdown]
# Notice that the computation time is much shorter than that of LOOCV. (In principle, the computation time for LOOCV for a least squares linear model should be faster than for K-fold CV, due to the availability of the formula
# $$\begin{aligned}
# \large CV_{(n)} = \frac {1} {n} \sum_{i=1}^n \Big ( \frac {y_i - \hat{y_i}} {1 - h_i}\Big )^2 \\
# \large \text { where } h_i = \frac {1} {n} + \frac {(x_i - \bar{x})^2} {\sum_{l=1}^n (x_l - \bar{x})^2 }
# \end{aligned}$$
# for LOOCV; however, the generic `cross_validate()` function does not make use of this formula.) We still see little evidence that using cubic or higher-degree polynomial terms leads to a lower test error than simply using a quadratic fit.

# %% [markdown]
# The `cross_validate()` function is flexible and can take different splitting mechanisms as an argument. For instance, one can use the `ShuffleSplit()` Shuffle function to implement the validation set approach just as easily as K-fold cross-validation.

# %%
validation = ShuffleSplit(n_splits =1, test_size =196, random_state =0)
results = cross_validate(hp_model, Auto.drop (['mpg'], axis=1), Auto['mpg'], cv=validation);
results['test_score'][0]

# %% [markdown]
# One can estimate the variability in the test error by running the following:

# %%
validation = ShuffleSplit(n_splits=10, test_size=196, random_state=0)
results = cross_validate(hp_model, Auto.drop (['mpg'], axis=1), Auto['mpg'], cv=validation)
results['test_score'].mean (), results['test_score'].std()

# %% [markdown]
# This standard deviation is not a valid estimate of the sampling variability of the mean test score or the individual scores, since the randomly-selected training samples overlap and hence introduce correlations. But it does give an idea of the Monte Carlo variation incurred by picking different random folds.

# %% [markdown]
# ## The Bootstrap

# %% [markdown]
# ### Estimating the Accuracy of a Statistic of Interest

# %% [markdown]
# The bootstrap approach can be applied in almost all situations. No complicated mathematical calculations are needed.
#
# To illustrate the bootstrap, we start with a simple example. The Portfolio data set in the `ISLP` package is described in Section 5.2 The Bootstrap. The goal is to estimate the sampling variance of the parameter $\alpha$ given in formula
# $$
# \large \hat{\alpha} = \frac {\hat{\sigma}_Y^2 - \hat{\sigma}_{XY}^2 } {\hat{\sigma}_X^2 + \hat{\sigma}_Y^2 - 2\hat{\sigma}_{XY}^2 }
# $$
#
# We will create a function `alpha_func()`, which takes as input a dataframe `D` assumed to have columns `X`and `Y`, as well as a vector `idx` indicating which observations should be used to estimate $\alpha$. The function then outputs the estimate for $\alpha$ based on the selected observations.

# %%
Portfolio = load_data('Portfolio')
print(Portfolio.head())
print(len(Portfolio))

def alpha_func(D, idx):
  cov_ = np.cov(D[['X','Y']]. loc[idx], rowvar=False)
  return (( cov_ [1,1] - cov_ [0 ,1]) / (cov_ [0 ,0]+ cov_ [1 ,1] -2* cov_ [0 ,1]))


# %% [markdown]
# This function returns an estimate for $\alpha$ based on applying the minimum variance formula (5.7) to the observations indexed by the argument `idx`. For instance, the following command estimates $\alpha$ using all 100 observations.

# %%
alpha_func(Portfolio , range(100))

# %% [markdown]
# Next we randomly select 100 observations from `range(100)`, with replacement. This is equivalent to constructing a new bootstrap data set and recomputing $\hat{\alpha}$ based on the new data set.

# %%
rng = np.random.default_rng(0)
alpha_func(Portfolio, rng.choice(100, 100, replace=True))


# %% [markdown]
# This process can be generalized to create a simple function `boot_SE()` for computing the bootstrap standard error for arbitrary functions that take only a data frame as an argument.

# %%
def boot_SE(func, D, n=None, B=1000, seed=0):
  rng = np.random.default_rng(seed)
  first_ , second_ = 0, 0
  n = n or D.shape [0]
  values = np.zeros(B)
  for i in range(B):
    idx = rng.choice(D.index, n, replace=True)
    value = func(D, idx)
    values[i] = value
  return np.std(values)


# %% [markdown]
# Let's use our function to evaluate the accuracy of our estimate of $\alpha$ using B = 1,000 bootstrap replications.

# %%
alpha_SE = boot_SE(alpha_func, Portfolio, B=1000, seed=0)
alpha_SE

# %%
printmd(f"The final output shows that the bootstrap estimate for SE(α̂) is {alpha_SE:.4f}.")

# %% [markdown]
# ### Estimating the Accuracy of a Linear Regression Model
#
# The bootstrap approach can be used to assess the variability of the coefficient estimates and predictions from a statistical learning method. Here we use the bootstrap approach in order to assess the variability of the estimates for $\beta_0$ and $\beta_1$ ,the intercept and slope terms for the linear regression model that uses horsepower to predict *mpg* in the **Auto** data set. We will compare the estimates obtained using the bootstrap to those obtained using the formulas for $SE(\beta_0)$ and $SE(\beta_1$) described in Section 3.1.2., Assessing the Accuracy of the Coefficients.
#
# $$\begin{aligned}
# \large & {SE(\hat{\beta_0})}^2 = \sigma^2 \Big [ \frac {1} {n} + \frac {{\bar{x}}^2}  {\sum_{i=1}^n(x_i - \bar{x})^2} \Big ] \\
# \large & {SE(\hat{\beta_1})}^2 =  \frac {\sigma^2}  {\sum_{i=1}^n(x_i - \bar{x})^2} \\
# \large & \text{ where } \sigma^2 = Var(\epsilon) \\
# \large & \text{ and } \hat{\sigma} = RSE = \sqrt{\frac {RSS} {n-2}}
# \end{aligned}$$

# %% [markdown]
# To use our `boot_SE()` function, we must write a function (its first argument) that takes a data frame `D` and indices `idx` as its only arguments. But here we want to bootstrap a specific regression model, specified by a model formula and data. We show how to do this in a few simple steps.

# %% [markdown]
# We start by writing a generic function `boot_OLS()` for bootstrapping a regression model that takes a formula to define the corresponding regression. We use the `clone()` function to make a copy of the formula that can be refit to the new dataframe. This means that any derived features such as those defined by `poly()` (which we will see shortly), will be re-fit on the resampled data frame.

# %%
allDone();
