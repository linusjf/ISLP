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

# %%
from GenExact import *
import numpy as np

# %%
# Define a vector of means and a matrix of covariances
mean = np.array([3, 3])
Sigma = np.array([[1, 0.70],
           [0.70, 1]])
gen_exact(mean=mean,sigma=Sigma,size=(100))[1]

# %%
