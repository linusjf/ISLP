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
# # Exercise 14: This problem focuses on the collinearity problem.

# %% [markdown]
# ## Import notebook funcs

# %%
from notebookfuncs import *

# %% [markdown]
# ## Import user funcs

# %%
from userfuncs import *

# %% [markdown]
# ## Import libraries

# %%
from sympy import symbols, poly
import numpy as np
import seaborn as sns
import pandas as pd
import statsmodels.formula.api as smf

# %% [markdown]
# ## (a) Perform the following commands in Python:

# %% [markdown]
# ```python
# rng = np.random.default_rng (10)
# x1 = rng.uniform (0, 1, size =100)
# x2 = 0.5 * x1 + rng.normal(size =100) / 10
# y = 2 + 2 * x1 + 0.3 * x2 + rng.normal(size =100)
# ```
# *The last line corresponds to creating a linear model in which $y$ is a function of $x_1$ and $x_2$. Write out the form of the linear model. What are the regression coefficients?*

# %%
x1, x2, y = symbols("x_1 x_2 y")
beta_0, beta_1, beta_2 = symbols(r"\beta_0 \beta_1 \beta_2")
equation =  beta_0 +  beta_1 * x1 + beta_2 * x2
display(equation)
equation = equation.subs([(beta_0,2), (beta_1,2),(beta_2, 0.3)])
equation = poly(equation)

# %%
rng = np.random.default_rng (10)
x1 = rng.uniform (0, 1, size =100)
x2 = 0.5 * x1 + rng.normal(size =100) / 10
y = 2 + 2 * x1 + 0.3 * x2 + rng.normal(size =100);

# %% [markdown]
# ## (b) What is the correlation between $x_1$ and $x_2$? Create a scatterplot displaying the relationship between the variables.

# %% [markdown]
# #### Correlation between x1 and x2

# %%
np.corrcoef(x1,x2)[0][1]


# %% [markdown]
# #### Display scatterplot of x1 against x2

# %%
def construct_df(x1, x2,y):
  df = pd.DataFrame({"x1": x1,"x2": x2, "y": y})
  return df

df = construct_df(x1,x2,y)
sns.scatterplot(df, x="x1", y="x2");


# %% [markdown]
# ## (c) Using this data, fit a least squares regression to predict y using $x_1$ and $x_2$. Describe the results obtained. What are $\beta_0$, $\beta_1$ , and $\beta_2$ ? How do these relate to the true $\beta_0$, $\beta_1$ , and $\beta_2$ ? Can you reject the null hypothesis $H_0 : \beta_1 = 0$? How about the null hypothesis $H_0 : \beta_2 = 0$?

# %% [markdown]
# ### Fit a least squares regression

# %%
# Fit combined regression
def fit_combined(df):
  formula = "y ~ x1 + x2"
  model = smf.ols(f"{formula}", df)
  results = model.fit()
  print(results.summary())
  return results

results = fit_combined(df);

# %% [markdown]
# ### Describe the results

# %% [markdown]
# - The regression tests whether the coefficients $\beta_0 \:, \beta_1 \: and \: \beta_2$ are 0. This is the null hypothesis.
# - From the p-values, we deduce that the intercept and $\beta_1$ are significant and hence we do not accept the null hypothesis for them.
# - $\beta_2$ , however, is not significant and thus its null hypothesis is accepted.
# - The adjusted $R^2$ is 0.276 i.e., 27.6% of the variance of the response (y) is explianed by the regressors x1 and x2.

# %% [markdown]
# ### What are $\beta_0$, $\beta_1$ and $\beta_2$?

# %%
params = results.params.to_frame().transpose()
params["Index"] = ["Estimate"]
params.set_index("Index")

# %% [markdown]
# ### How do these relate to the true $\beta_0$, $\beta_1$ and $\beta_2$?

# %%
coeffs = equation.coeffs()
orig = pd.DataFrame({"Intercept": coeffs[2], "x1": coeffs[0], "x2": coeffs[1]}, index=[0])
orig["Index"] = ["Original"]
orig.set_index("Index")
res = pd.concat([params,orig], axis=0).set_index("Index")

# %% [markdown]
# ### Influential points

# %%
get_influence_points(results)


# %% [markdown]
# ## (d) Now fit a least squares regression to predict y using only $x_1$. Comment on your results. Can you reject the null hypothesis $H_0 : \beta_1 = 0$?

# %%
def fit_x1(df):
  formula = "y ~ x1"
  model = smf.ols(f"{formula}", df)
  results = model.fit()
  print(results.summary())
  return results

results = fit_x1(df);

# %% [markdown]
# ### Can you reject the null hypothesis $H_0 : \beta_1 = 0$?

# %% [markdown]
# - Yes, we can reject the null hypothesis $H_0 : \beta_1 = 0$  since the p-value for the coefficient of $x_1$ is significant.

# %% [markdown]
# ### Influential points

# %%
get_influence_points(results)


# %% [markdown]
# ## (e) Now fit a least squares regression to predict y using only $x_2$. Comment on your results. Can you reject the null hypothesis $H_0 : \beta_1 = 0$?

# %%
def fit_x2(df):
  formula = "y ~ x2"
  model = smf.ols(f"{formula}", df)
  results = model.fit()
  print(results.summary())
  return results

results = fit_x2(df);

# %% [markdown]
# ### Can you reject the null hypothesis $H_0 : \beta_1 = 0$?

# %% [markdown]
# - Yes, we can reject the null hypothesis $H_0 : \beta_1 = 0$  since the p-value for the coefficient of $x_2$ is significant.

# %% [markdown]
# ### Influential points

# %%
get_influence_points(results)

# %% [markdown]
# ## (f) Do the results obtained in (c)–(e) contradict each other? Explain your answer.

# %% [markdown]
# - No, the results do not contradict each other since the two variables are collinear and contain the same information. 
# - Thus, they can be interchanged for each other without much loss of information in the regression model.

# %% [markdown]
# ## (g) Suppose we obtain one additional observation, which was unfortunately mismeasured. We use the function np.concatenate() to add this additional observation to each of $x_1, x_2 \: and \: y$.
# ```python
# x1 = np.concatenate ([x1 , [0.1]])
# x2 = np.concatenate ([x2 , [0.8]])
# y = np.concatenate ([y, [6]])
# ```
# ## Re-fit the linear models from (c) to (e) using this new data. What effect does this new observation have on the each of the models? In each model, is this observation an outlier? A high-leverage point? Both? Explain your answers.
#

# %% [markdown]
# ### Add an additional observation

# %%
x1 = np.concatenate ([x1 , [0.1]])
x2 = np.concatenate ([x2 , [0.8]])
y = np.concatenate ([y, [6]]);

# %%
x1[-1], x2[-1], y[-1]

# %%
df = construct_df(x1,x2,y)
df.tail(1)

# %% [markdown]
# ### Combined regression

# %%
results = fit_combined(df);

# %% [markdown]
# - Here, we see the effect of the additional mismeasured data point.
# - The effect on the combined regression is to switch the significance of the regressors x1 and x2.
# - Now, the coefficient of x1 is not statistically significant with a p-value of 0.07. 

# %% [markdown]
# ### Residuals, outliers, leverage and influence

# %%
display_cooks_distance_plot(results);

# %%
display_hat_leverage_plot(results)

# %%
get_influence_points(results)

# %% [markdown]
# - From the above, we can see that there are two influential datapoints, 99 ans 100.
# - This is initially surprising until we compute the influential points without the freshly added mismeasured data point and discover that point 99 was influential in the earlier regression.

# %% [markdown]
# ### Regress on x1

# %%
results = fit_x1(df);

# %%
display_cooks_distance_plot(results);

# %%
display_hat_leverage_plot(results)

# %%
get_influence_points(results)

# %% [markdown]
# - Similarly, in the regression of y on x1 only, we find points 99 and 100 to be influential.

# %% [markdown]
# ### Regress on x2

# %%
results = fit_x2(df);

# %%
display_cooks_distance_plot(results);

# %%
display_hat_leverage_plot(results)

# %%
get_influence_points(results)

# %% [markdown]
# - In the regression of y on x2, no data point is influential since neither the studentized residuals or their associated p-values cross the thresholds for these parameters. 

# %%
allDone();
