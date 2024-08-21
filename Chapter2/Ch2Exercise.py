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
import matplotlib.pyplot as plt
from matplotlib.pyplot import subplots
import pandas as pd

# %%
College = pd.read_csv("College.csv")
College

# %%
college2 = pd.read_csv("College.csv", index_col=0)
college2

# %%
College3 = College.rename({"Unnamed: 0": "College"}, axis=1)
College3.set_index("College")
College3

# %%
College = College3

# %%
College.describe()

# %%
pd.plotting.scatter_matrix(College[["Top10perc", "Apps", "Enroll"]]);

# %%
fig, ax = subplots(figsize = (8,8))
College.boxplot("Outstate", by = "Private", ax = ax);

# %%
College["Top10perc"]

# %%
College["Elite"] = pd.cut(College["Top10perc"], [0,50, 100], labels=["No","Yes"])
College["Elite"].value_counts()


# %%
fig, ax = subplots(figsize = (8, 8))
College.boxplot("Outstate", by = "Elite", ax = ax);

# %%
College.plot.hist();

# %%
fig , ax = subplots(figsize=(8,8))
College.hist("Apps", ax = ax);

# %%
numeric_columns = College.select_dtypes(include="number").columns.tolist()
numeric_columns

# %%
fig, axs = subplots(4, 4, figsize = (16,16))
for row in range(0,4):
  for column in range(0,4):
    College.hist(numeric_columns[row * 4 + column], ax = axs[row,column]);
# %%
