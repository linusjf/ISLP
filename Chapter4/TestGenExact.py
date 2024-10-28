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
# ## Import notebook functions

# %%
from notebookfuncs import *

# %%
from GenExact import *
import numpy as np

# %%
# Define a vector of means and a matrix of covariances
mean = np.array([3, 3])
Sigma = np.array([[1, 0.70],
           [0.70, 1]])

# %%
rng = np.random.RandomState(0)
x1 = gen_exact(mean=mean,sigma=Sigma,size=(100),rng=rng);

# %%
np.cov(x1,rowvar=False,bias=True)

# %%
np.mean(x1)

# %%
rng = np.random.RandomState(0)
x2 = gen_inexact(mean=mean,sigma=Sigma,size=(100),rng=rng);

# %%
np.cov(x2,rowvar=False,bias=True)

# %%
np.mean(x2)

# %%
np.allclose(x1,x2)

# %%
allDone();
