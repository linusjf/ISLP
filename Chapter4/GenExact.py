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
import numpy as np

def gen_exact(mean=None,sigma=None,size=None):
  if (mean is None or sigma is None or size is None):
    return None
  # Generate size cases
  X = np.random.default_rng(0).multivariate_normal(mean, sigma, size=size).T

  # Subtract the mean from each variable
  for n in range(X.shape[0]):
      X[n] = X[n] - X[n].mean()

  # Make each variable in X orthogonal to one another
  L_inv = np.linalg.cholesky(np.cov(X, bias = True))
  L_inv = np.linalg.inv(L_inv)
  X = np.dot(L_inv, X)

  # Rescale X to exactly match Sigma
  L = np.linalg.cholesky(sigma)
  X = np.dot(L, X)

  # Add the mean back into each variable
  for n in range(X.shape[0]):
      X[n] = X[n] + mean[n]

  # The covariance of the generated data should match Sigma
  cov = np.cov(X, bias = True)
  X = X.T
  return X, cov





# %%

# %%
