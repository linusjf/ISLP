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
Boston_quant = Boston.drop("chas", axis = 1)

# %%
print(np.unique(Boston_quant["zn"]))
median_medv = Boston_quant.groupby(["zn"], observed=True)[["medv"]].median()
median_medv

# %%
from numpy import median
sns.catplot(data=Boston_quant, x="zn", y="medv", kind="bar", height = 10, aspect = 2, estimator = median);

# %%
print(np.unique(Boston_quant["rad"]))
mean_rad = Boston_quant.groupby(["rad"], observed=True)[["medv"]].mean()
mean_rad

# %%
sns.catplot(data=Boston_quant, x="rad", y="medv", kind="bar", height = 10, aspect = 2);

# %%
import seaborn as sns
sns.set_theme(style="ticks")
g = sns.pairplot(Boston_quant, height = 5, aspect = 2, diag_kind = "kde", y_vars=["medv"]);

# %% [markdown]
# Plotting the other quantitative columns against medv (Median value of owner-occupied homes), we can see that:
# 1. crim is negatively correlated with medv. i.e., as crime rate increases, median value of homes decrease.
# 2. indus is negatively correlated with medv which is expected as industrialisation of a town increases, the house prices decrease.
# 3. nox is negatively correlated with medv which is also expected.
# 4. as the number of rooms (rm) increase, so does the value of the home.
# 5. as the proportion of homes built prior to 1940 increase, the value of homes in that area decrease. There are some notable outliers, but that appears to be the general trend.
# 6. There is a clear relationship in the lsat (lower status of population percent) versus medv where medv decreases with the increase in lstat on the x-axis.

# %%
Boston_quant["zn"].value_counts()

# %%
import seaborn as sns
sns.set_theme(style="ticks")
g = sns.pairplot(Boston_quant, height = 5, aspect = 2, diag_kind = "kde", y_vars=["crim"]);

# %%
sns.catplot(data=Boston_quant, x="zn", y="crim", kind="bar", height = 6, aspect = 2);

# %%
sns.displot(data=Boston_quant, x="indus", y="crim",height = 4, aspect = 3);

# %%
sns.displot(data=Boston_quant, x="age", y="crim",height = 4, aspect = 3);

# %%
sns.displot(data=Boston_quant, x="nox", y="crim",height = 4, aspect = 3);

# %%
sns.displot(data=Boston_quant, x="dis", y="crim",height = 4, aspect = 3);

# %%
sns.displot(data=Boston_quant, x="tax", y="crim",height = 4, aspect = 3);

# %%
sns.displot(data=Boston_quant, x="ptratio", y="crim",height = 4, aspect = 3);

# %% [markdown]
# We've already seen that there appears to be a relationship b/w crime rate and medv where a higher crime rate is associated with lower property prices.
# Additionally, plotting the other quantitative variables against crime rate (crim), we can perceive the following:
# 1. No-zoned areas or towns are associated with higher crime rate compared to all other zoning percentages.
# 2. For some reason, industrialization of around 18% displays a spike in the crime rate compared to the other suburbs. This might be worth investigating further.
# 3. Suburbs with nox > 0.55 or so have an elevated crime rate. That could be because lower strata income people live in those areas, and they are more inclined to criminal activities.
# 4. There also seems to be an increasing relationship b/w crime rate and percentage  of homes built prior to 1940. Once that percentage crosses 40%, there is an increasing number of suburbs that exhibit elevated crime rates.
# 5. Suburbs within a distance to Boston employment centres that range from 1 to 4.5 show an elevated crime rate. This needs to be investigated further. Where are these employment centres located?
# 6. There seems to be a higher incidence of crimes for areas with tax rate around 670. Why? 
# 7. The crime rate does not seem to have a strong relationship with ptratio, but for around point 20.1 where the crime rate spikes compared to the other areas. 
# 8. Crime rate decreases as the median value of properties rise across suburbs as a whole.

# %% [markdown]
#  *Do any of the suburbs of Boston appear to have particularly high crime rates? Tax rates? Pupil-teacher ratios? Comment on the range of each predictor.*

# %% [markdown]
# *How many of the suburbs in this data set bound the Charles River?*

# %%
*What is the median pupil-teacher ratio among the towns in this data set?*

# %%
*Which suburb of Boston has lowest median value of owner-occupied homes? What are the values of the other predictors for that suburb, and how do those values compare to the overall ranges for those predictors? Comment on your findings.*

# %%
*In this data set, how many of the suburbs average more than seven rooms per dwelling? More than eight rooms per dwelling? Comment on the suburbs that average more than eight rooms per dwelling.*
