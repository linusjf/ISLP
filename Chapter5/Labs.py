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

# %%
allDone();
