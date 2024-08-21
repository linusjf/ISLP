# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
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
# # Chapter 2 Lab

# %%
print("Fit a model with ", 11 , " variables")


# %%
# print?

# %%
3 + 5

# %%
"hello, " + " " + "world!"

# %%
x = [3,4,5]
x

# %%
y = [4,9,7]
x + y

# %%
import numpy as np

# %%
x = np.array([3,4,5])
y = np.array([4,9,7])
x + y

# %%
x = np.array([[1,2], [3,4]])
x

# %%
x.ndim


# %%
x.dtype

# %%
x = np.array([[1,2], [3.0,4]])
x.dtype

# %%
np.array ([[1, 2], [3, 4]], float).dtype

# %%
x.shape

# %%
x = np.array([1,2,3,4])
x.sum()

# %%

# %%
x = np.array([1,2,3,4])
np.sum(x)

# %%
x = np.array([1,2,3,4,5,6])
print("Beginning x:\n ", x)
x_reshape = x.reshape(2,3)
print("reshaped x:\n", x_reshape)

# %%
x_reshape[0,0]

# %%
x_reshape[1,2]

# %%
print("x before we modify x_reshape:\n", x)
print("x_reshape before we modify x_reshape:\n", x_reshape)
x_reshape[0,0] = 5
print("x_reshape after we modify its top left element:\n", x_reshape)
print("x after we modify top left element of x_reshape:\n", x)


# %%
my_tuple = (1,2,3)
# type error
# my_tuple[0] = 10

# %%
x_reshape.shape, x_reshape.ndim, x_reshape.T

# %%
np.sqrt(x)

# %%
x**2

# %%
x**0.5

# %%
# np.random.normal?

# %%
x = np.random.normal(size=50)
x

# %%
y = x + np.random.normal(loc=50, scale=1, size=50)

# %%
np.corrcoef(x,y)

# %%
print(np.random.normal(scale=5, size=2))
print(np.random.normal(scale=5, size=2))

# %%
rng = np.random.default_rng(1303)
print(rng.normal(scale=5, size=2))
rng = np.random.default_rng(1303)
print(rng.normal(scale=5, size=2))

# %%
rng = np.random.default_rng(3)
y = rng.standard_normal(10)
np.mean(y), y.mean()

# %%
np.var(y), y.var(), np.mean((y - y.mean())**2)

# %%
np.sqrt(np.var(y)), np.std(y)

# %%
# np.var?

# %%
X = rng.standard_normal((10,3))
X

# %%
X.mean(axis=0)

# %%
X.mean(axis=0)

# %%
X.mean(0)

# %%
X.mean(1)

# %%
X.mean()

# %%
# ax.plot?

# %%
from matplotlib.pyplot import subplots

# %%
fig, ax = subplots(figsize=(8,8))
x = rng.standard_normal(100)
y = rng.standard_normal(100)
ax.plot(x, y);

# %%
fig, ax = subplots(figsize=(8,8))
ax.plot(x, y, 'o');

# %%
fig, ax = subplots(figsize=(8,8))
ax.scatter(x, y, marker =  'o');

# %%
fig, ax = subplots(figsize=(8,8))

ax.scatter(x, y, marker = 'o');
ax.set_xlabel("this is the x-axis")
ax.set_ylabel("this is the y-axis")
ax.set_title("Plot of X vs Y");

# %%
fig.set_size_inches(12, 3)
fig

# %%
fig, axes = subplots(nrows = 2, ncols = 3, figsize=(15,5))

# %%
axes[0,1].plot(x, y, 'o')
axes[1, 2].scatter(x, y, marker='+');
fig

# %%
fig.savefig("Figure.png", dpi=400)
fig.savefig("Figure.pdf", dpi=200);

# %%
axes[0, 1].set_xlim([-1, 1])
fig.savefig("Figure_updated.jpg")
fig

# %%
fig, ax = subplots(figsize = (8,8))
x = np.linspace(-np.pi, np.pi, 50)
print(x)
y = x
f = np.multiply.outer(np.cos(y), 1 / (1 + x**2))
ax.contour(x, y, f);

# %%
fig, ax = subplots(figsize = (8,8))
ax.contour(x, y, f, levels = 45);

# %%
fig, ax = subplots(figsize = (8,8))
ax.imshow(f);

# %%
seq1 = np.linspace(0,10, 11)
seq1

# %%
seq2 = np.arange(0, 10)
seq2

# %%
"Hello, world!"[3:6]

# %%
"Hello, world!"[slice(3,6)]

# %%
A = np.array(np.arange(16)).reshape((4,4))
print(A)
A[1,2]

# %%
A[[1,3]]

# %%
A[:,[0,2]]

# %%
A[[1,3],[0,2]]

# %%
np.array([A[1,0],A[3,2]])

# %%
A[[1,3]][:,[0,2]]

# %%
idx = np.ix_([1,3],[0,2,3])
A[idx]

# %%
A[1:4:2,0:3:2]

# %%
keep_rows = np.zeros(A.shape[0], bool)
keep_rows

# %%
keep_rows[[1,3]] = True
keep_rows

# %%
np.all(keep_rows == np.array([0, 1,0,1]))

# %%
A[np.array([0,1,0,1])]

# %%
A[keep_rows]

# %%
keep_cols = np.zeros(A.shape[1], bool)
keep_cols

# %%
keep_cols[[0,2,3]] = True
keep_cols

# %%
idx_bool = np.ix_(keep_rows, keep_cols)
idx_bool

# %%
A[idx_bool]

# %%
idx_mixed = np.ix_([1,3], keep_cols)
idx_mixed

# %%
A[idx_mixed]

# %% [markdown]
# ## Reading in a data set

# %%
import pandas as pd
Auto = pd.read_csv("Auto.csv")
Auto

# %%
Auto = pd.read_csv("Auto.data", sep="\s+")
Auto

# %%
Auto["horsepower"]

# %%
np.unique(Auto["horsepower"])

# %%
Auto = pd.read_csv("Auto.data", na_values=["?"], sep="\s+")
Auto["horsepower"].sum()

# %%
Auto.shape

# %%
Auto_new = Auto.dropna()

# %%
Auto_new.shape

# %%
Auto = Auto_new
Auto.columns

# %%
Auto[:3]

# %%
idx_80 = Auto['year'] > 80
Auto[idx_80]

# %%
Auto[["mpg","horsepower"]]

# %%
Auto.index

# %%
Auto_re = Auto.set_index("name")
Auto_re

# %%
Auto_re.columns

# %%
Auto_re.shape

# %%
rows = ["amc rebel sst", "ford torino"]
Auto_re.loc[rows]

# %%
Auto_re.iloc[[3,4]]

# %%
Auto_re.iloc[:,[0,2,3]]

# %%
Auto_re.iloc[[3,4],[0,2,3]]

# %%
Auto_re.loc['ford galaxie 500', ["mpg", "origin"]]

# %%
idx_80 = Auto_re["year"] > 80
Auto_re.loc[idx_80]

# %%
Auto_re.loc[idx_80, ["weight", "origin"]]

# %%
Auto_re.loc[lambda df: df['year'] > 80, ["weight", "origin"]]

# %%
Auto_re.loc[lambda df: (df["year"] > 80) & (df["mpg"] > 30), ["weight", "origin"]]

# %%
Auto_re.loc[lambda df: (df["displacement"] < 300)
& (df.index.str.contains("ford")
| df.index.str.contains("datsun")),
  ["weight", "origin"]
     ]

# %% [markdown] jupyter={"outputs_hidden": true}
# ## for loops

# %%
total = 0
for value in [3,2,9]:
  total += value
print("total is: {0}".format(total))

# %%
total = 0
for value in [3,2,9]:
  for weight in [3,2,1]:
    total += weight * value
print("total is: {0}".format(total))

# %%
total = 0
for value, weight in zip([3,2,9], [0.2,0.3,0.5]):
  total += weight * value
print("weighted average is: {0}".format(total))

# %%
rng = np.random.default_rng(1)
A = rng.standard_normal((127, 5))
A

# %%
A.shape


# %%
M = rng.choice([0, np.nan], p = [0.8, 0.2], size=A.shape)
M

# %%
A += M
A

# %%
D = pd.DataFrame(A, columns = ["food",
                               "bar",
                               "pickle",
                               "snack",
                               "popcorn"
                              ])
D

# %%
D[:3]

# %%
for col in D.columns:
  template = "Column {0} has {1: .2%} missing values"
  print(template.format(col, np.isnan(D[col]).mean()))

# %%
fig, ax = subplots(figsize=(8,8))
ax.plot(Auto["horsepower"].values, Auto["mpg"].values, 'o');

# %%
ax = Auto.plot.scatter("horsepower", "mpg")
ax.set_title("Horsepower vs MPG");

# %%
fig = ax.figure
fig.savefig("hp_mpg.png")

# %%
fig, axes = subplots(ncols=3, figsize=(15,5))
Auto.plot.scatter("horsepower","mpg", ax=axes[1]);

# %%
Auto.cylinders = pd.Series(Auto.cylinders, dtype="category")
Auto.cylinders.dtype

# %%
fig, ax = subplots(figsize=(8,8))
Auto.boxplot("mpg", by="cylinders",ax=ax);

# %%
fig, ax = subplots(figsize=(8,8))
Auto.hist("mpg", ax=ax);

# %%
fig, ax = subplots(figsize=(8,8))
Auto.hist("mpg", color="red", bins=12, ax=ax);

# %%
# Auto.hist?

# %%
pd.plotting.scatter_matrix(Auto);

# %%
pd.plotting.scatter_matrix(Auto[["mpg", "displacement", "weight"]]);

# %%
Auto[["mpg","weight"]].describe()

# %%
Auto[["cylinders"]].describe()

# %%
Auto[["mpg"]].describe()

# %%
