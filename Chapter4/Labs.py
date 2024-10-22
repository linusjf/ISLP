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
# ## Examine the Smarket data --- part of the ISLP library.

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
Smarket

# %%
Smarket.columns

# %%
Smarket.Direction = Smarket.Direction.astype("category")

# %%
Smarket.corr(numeric_only=True)

# %%
allDone();
