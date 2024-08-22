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
# %% [markdown]
# ### Count of private and public colleges

# %%
College["Private"].value_counts()

# %%
College["AcceptanceRate"] = round(College["Accept"]/College["Apps"] * 100,2)
College["AcceptanceRate"]

# %%
### Plot boxplot for acceptance rate by College Type : Elite or not

# %%
fig, ax = subplots(figsize=(8,8))
College.boxplot("AcceptanceRate", by = "Elite", ax = ax);

# %%
### Plot boxplot for acceptance rate for Private colleges or not

# %%
fig, ax = subplots(figsize=(8,8))
College.boxplot("AcceptanceRate", by = "Private", ax = ax);

# %%
College["EnrollmentRate"] = round(College["Enroll"]/College["Accept"] * 100, 2)
College["EnrollmentRate"]

# %%
fig, ax = subplots(figsize=(8,8))
College.boxplot("EnrollmentRate", by = "Elite", ax = ax);

# %%
fig, ax = subplots(figsize=(8,8))
College.boxplot("EnrollmentRate", by = "Private", ax = ax);

# %%
College["StudentCosts"] = College["Outstate"] + College["Room.Board"] + College["Books"] + College["Personal"]

# %%
fig, ax = subplots(figsize=(8,8))
College.boxplot("StudentCosts", by = "Elite", ax = ax);

# %%
fig, ax = subplots(figsize=(8,8))
College.boxplot("StudentCosts", by = "Private", ax = ax);

# %%
fig, ax = subplots(figsize=(8,8))
College.boxplot("PhD", by = "Private", ax = ax);

# %%
fig, ax = subplots(figsize=(8,8))
College.boxplot("PhD", by = "Elite", ax = ax);

# %%
fig, ax = subplots(figsize=(8,8))
College.boxplot("Terminal", by = "Elite", ax = ax);

# %%
fig, ax = subplots(figsize=(8,8))
College.boxplot("Terminal", by = "Private", ax = ax);

# %%
fig, ax = subplots(figsize=(8,8))
College.boxplot("S.F.Ratio", by = "Private", ax = ax);

# %%
fig, ax = subplots(figsize=(8,8))
College.boxplot("S.F.Ratio", by = "Elite", ax = ax);

# %%
fig, ax = subplots(figsize=(8,8))
College.boxplot("perc.alumni", by = "Elite", ax = ax);

# %%
fig, ax = subplots(figsize=(8,8))
College.boxplot("perc.alumni", by = "Private", ax = ax);

# %%
fig, ax = subplots(figsize=(8,8))
College.boxplot("Expend", by = "Private", ax = ax);

# %%
fig, ax = subplots(figsize=(8,8))
College.boxplot("Expend", by = "Elite", ax = ax);

# %%
fig, ax = subplots(figsize=(8,8))
College.boxplot("Grad.Rate", by = "Elite", ax = ax);

# %%
fig, ax = subplots(figsize=(8,8))
College.boxplot("Grad.Rate", by = "Private", ax = ax);

# %%
mean_grad_rate = College.groupby("Elite", observed=True)[["AcceptanceRate","EnrollmentRate","Grad.Rate"]].mean()
mean_grad_rate

# %%
mean_grad_rate.plot(y = ["AcceptanceRate","EnrollmentRate","Grad.Rate"],kind="bar", rot=0);

# %%
