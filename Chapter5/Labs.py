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
# We now use the `predict()` method of results evaluated on the model matrix for this model created using the validation data set. We also calculate the validation MSE of our model.

# %%
X_valid = hp_mm.transform(Auto_valid)
y_valid = Auto_valid['mpg']
valid_pred = results.predict(X_valid)
mse = np.mean((y_valid - valid_pred)**2)

# %%
printmd(f"Hence our estimate for the validation MSE of the linear regression fit is {mse: 0.2f}.")

# %%
allDone();
