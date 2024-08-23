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
Auto = pd.read_csv("Auto.csv", na_values={"?"})
print(Auto.shape)
np.unique(Auto["horsepower"])

# %%
### Which predictors are quantitative and which are qualitative?

# %% [markdown]
# Rename the misleading column name acceleration to timetoacceleration since it's a tad misleading.

# %%
Auto["timetoacceleration"] = Auto["acceleration"]
Auto = Auto.drop("acceleration", axis = 1)

# %%
Auto = Auto.dropna()
Auto.shape

# %%
Auto.describe()

# %%
Auto["origin"] = Auto.origin.astype("category")
Auto["year"] = Auto.year.astype("category")
Auto["cylinders"] = Auto.cylinders.astype("category")
print(np.unique(Auto["year"]))
print(np.unique(Auto["cylinders"]))

# %%
Auto["origin"] = Auto["origin"].cat.rename_categories({1: "American",2: "European",3:"Japanese"})
np.unique(Auto["origin"])

# %%
Auto.head()

# %%
Auto = Auto.set_index("name")

# %%
Auto

# %%
Auto_new = Auto.drop(Auto.index[10:86])
Auto_new.describe()

# %%
Auto_new

# %% [markdown]
# Using the full data set, investigate the predictors graphically, using scatter plots or other tools of your choice. Create some plots highlighting the relationships among the predictors. Comment on your findings.

# %%
pd.plotting.scatter_matrix(Auto, figsize=(14,14));

# %% [markdown]
# ### Findings:
# 1. Weight and displacement seem to be negatively correlated with MPG.
# 2. timetoacceleration (0–60 mph in seconds) seems to be positively correlated with MPG. As time to acceleration increases, MPG also increases. The longer the time to acceleration, the better the fuel efficiency.
# 3. weight is also positively correlated with displacement. As weight increases, so does displacement, i.e., as the body weight increases, so does displacement need to increase.
# 4. Displacement is seen to increase as the number of cylinders increase. This is expected since displacement is a function of the number of cylinders, amongst other components.
#
# We can conclude that MPG can be predicted using the variables weight, displacement and timetoacceleration.

# %%
mean_mpg_origin = Auto.groupby(["origin"], observed=True)[["mpg"]].mean()
mean_mpg_origin

# %%
mean_mpg_year = Auto.groupby(["year"], observed=True)[["mpg"]].mean()
mean_mpg_year

# %%
mean_mpg_cylinders = Auto.groupby(["cylinders"], observed=True)[["mpg"]].mean()
mean_mpg_cylinders

# %% [markdown]
# We can also observe that fuel efficiency is affected by the make of the car. Japanese > European > American
# The year also plays a significant role. Later model cars are more fuel efficient than the earlier models. Cars are also more fuel efficient with lesser number of cylinders. These can also be used as predictors to deduce the MPG. 

# %%
from ISLP import load_data
Boston = load_data('Boston')
Boston.columns

# %%
Boston.shape

# %% [markdown]
# The rows represent data for 506 suburbs in Boston. The columns represent housing values and variables of interest that may predict housing values in each suburb.

# %%
Boston.describe()

# %%
pd.plotting.scatter_matrix(Boston.drop("chas", axis = 1), figsize=(20,25));

# %%
