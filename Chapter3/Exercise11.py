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
# # Exercise 11

# %% [markdown]
# ## Import notebook funcs

# %%
from notebookfuncs import *

# %% [markdown]
# ## In this problem we will investigate the t-statistic for the null hypothesis $H_{0} : \beta = 0$ in simple linear regression without an intercept. To begin, we generate a predictor x and a response y as follows.
#
# ```python
# rng = np.random.default_rng (1)
#
# x = rng.normal(size =100)
#
# y = 2 * x + rng.normal(size =100)
# ```

# %%
import numpy as np
import pandas as pd

## [[https://stats.stackexchange.com/a/382059/270877]]
def generate_data(mean=0.0, sd=1.0, inverse=False):
  N = 100
  b = 2.0
  rng = np.random.default_rng(1)
  x = rng.normal(size=N)
  noise = rng.normal(loc=mean,scale=sd, size=N)
  y = b * x + noise
  if (inverse):
    y = x;
    x = (1/b) * y  -  (1/b ** 2) * noise

  df = pd.DataFrame({"x": x,"y":y})
  return df

df = generate_data()
std_x = np.std(df["x"])
std_y = np.std(df["y"])
df.head()


# %% [markdown]
# ## (a) Perform a simple linear regression of y onto x, without an intercept. Report the coefficient estimate $\hat{\beta}$, the standard error of this coefficient estimate, and the t-statistic and p-value associated with the null hypothesis $H_{0} : \beta = 0$. Comment on these results. (You can perform regression without an intercept using the keywords argument intercept=False to ModelSpec().)

# %%
import statsmodels.formula.api as smf

def get_results_df(results):
  result_df = pd.DataFrame({"coefficient": results.params,
                          "se": results.bse,
                          "tstatistic": results.tvalues,
                          "p-value":results.pvalues,
                         "r-squared": results.rsquared,
                         "pearson_coefficient": np.sqrt(results.rsquared),
                            "rss": results.ssr,
                            "sd_residuals": np.sqrt(results.mse_resid)
                           })
  return result_df

formula = "y ~ x + 0"
model = smf.ols(f'{formula}', df)
results = model.fit()
result_df = get_results_df(results)
result_df

# %%
print("Calculated t-statistic:")
print(f"{results.params/results.bse}")

# %% [markdown]
# - The $\beta_{1}$ estimate is 1.976242 which is close to the actual value of 2.0 used while generating the data.
# - The standard error is 0.116948 which is low
# - The p-value of 6.231546$e^{-31}$ suggests a strong relationship between x and y of the form: $y = 1.976242 * x$

# %%
from statsmodels.graphics.regressionplots import plot_fit
plot_fit(results, "x");

# %% [markdown]
# ## (b) Now perform a simple linear regression of x onto y without an intercept, and report the coefficient estimate, its standard error, and the corresponding t-statistic and p-values associated with the null hypothesis $H_{0} : \beta = 0$. Comment on these results.

# %%
formula = "x ~ y + 0"
model = smf.ols(f'{formula}', df)
results = model.fit()
result_df = get_results_df(results)
result_df

# %%
print("Calculated t-statistic:")
print(f"{results.params/results.bse}")

# %% [markdown]
# - The $\beta_{1}$ estimate, in this case, is 0.375744 which is not as close as one would expect to 0.5
# - The p-value of 6.231546$e^{-31}$ signifies a strong relationship between x and y of the form $x = 0.375744 * y$
# - The t-statistics and, hence, the p-values are identical in both regressions.
# - We can also see that the $R^{2}$ and $\rho$ (Pearson coefficient) are identical in both regressions.

# %%
plot_fit(results, "y");

# %%
pearson_coefficient = result_df["pearson_coefficient"]

# %% [markdown]
# ## (c) What is the relationship between the results obtained in (a) and (b)?

# %% [markdown]
# ### Why should the estimates of the coefficient of $\beta_{1}$ differ from the expected value of 0.5 in the regression of $x \thicksim y + 0$?
#
# - The obvious answer is that the cause of the variation is because of the introduction of the error term (or residuals) that we introduced while generating y from x with a mean of 0 and a standard deviation of 1.
# - For x and y to be perfectly symmetrical, the standard deviation of the error terms would have to be zero.
# - We can check that with the regression below where we generate y with zero SD and regress y on x and then x on y.

# %%
df = generate_data(sd=0);
std_x_perfect = np.std(df["x"])
std_y_perfect = np.std(df["y"])
std_x_perfect,std_y_perfect

# %%
formula = "y ~ x + 0"
model = smf.ols(f'{formula}', df)
results = model.fit()
result_df = get_results_df(results)
result_df

# %%
formula = "x ~ y + 0"
model = smf.ols(f'{formula}', df)
results = model.fit()
result_df = get_results_df(results)
result_df

# %% [markdown]
# ### Now, the formula for $\beta_{1(y \: on\: x)}$ is $\rho * \large \frac {S_{y}} {S_{x}}$ and for $\beta_{1(x\: on \: y)}$ is $\rho * \large \frac {S_{x}} {S_{y}}$.

# %% [markdown]
# ### Using the formulae above:

# %%
beta_1_x_on_y = pearson_coefficient * std_x / std_y
beta_1_y_on_x = pearson_coefficient * std_y / std_x
beta_1_y_on_x, beta_1_x_on_y

# %% [markdown]
# ### Now, if Pearson coeffcient were 1.0, i.e., there is perfect correlation between x and y, then we can calculate the betas and find them to be reciprocals of each other.

# %%
beta_1_x_on_y = 1.0 * std_x_perfect / std_y_perfect
beta_1_y_on_x = 1.0 * std_y_perfect / std_x_perfect
beta_1_y_on_x, beta_1_x_on_y

# %% [markdown]
# ### If we were to standardize x and y, we would have the following regression results

# %%
from scipy import stats
df = generate_data()
df["x"] = stats.zscore(df["x"])
df["y"] = stats.zscore(df["y"])
df.head()

# %%
formula = "y ~ x + 0"
model = smf.ols(f'{formula}', df)
results = model.fit()
result_df = get_results_df(results)
result_df

# %%
plot_fit(results, "x");

# %%
formula = "x ~ y + 0"
model = smf.ols(f'{formula}', df)
results = model.fit()
result_df = get_results_df(results)
result_df

# %%
plot_fit(results, "y");

# %% [markdown]
# ### This actually matches our intuition that the slopes of both lines converge and the coefficient is the same in both regressions. This is because the standardized x and y variables have standard deviation of 1 and hence the slope in either case is simply $\rho$.

# %% [markdown]
# ### If we try to mimimize the residuals using the formula $\frac {1} {b^2} * E(Y - bX)^2$

# %% [markdown]
# ### Using some data manipulation in  the generate_data method, we can come close to the expected estimate of 0.5 in the X ~ Y + 0 regression.

# %%
df = generate_data(inverse=True);

# %%
formula = "x ~ y + 0"
model = smf.ols(f'{formula}', df)
results = model.fit()
result_df = get_results_df(results)
result_df

# %% [markdown]
# Note:- The standard deviation of the residuals is $\frac {1} {b}^{2}$ times thr SD of the residuals for Y ~ X + 0.

# %%
plot_fit(results, "y");

# %% [markdown]
# ### If we again invert the regression, regression y on x, we have:

# %%
formula = "y ~ x + 0"
model = smf.ols(f'{formula}', df)
results = model.fit()
result_df = get_results_df(results)
result_df

# %% [markdown]
# - *Again, we see that since we're minimizing the distance between x and its fitted values in the first regression and y and its fitted values in the next, the estimatesof the first regression are close to the actual value but not so in its inverse.
# This only works if the residuals are manipulated to have SD of $\frac {-1} {b^2}$ in the second.*
# - *That's the most I can explain.*

# %% [markdown]
# ### We can check that the smaller the noise or residuals, the more likely that the regression of y on x and x on y are more or less reciprocals of each other in terms of the coefficient of the regressor.

# %%
from sklearn.linear_model import LinearRegression

# Generate data
np.random.seed(0)
x = np.random.rand(100)
denominator = 1.0
noise = np.random.randn(100)

while (True):

  print(f"SD of noise: {np.std(noise)}")
  y = 2 * x + noise
  # Fit regressions
  x_on_y = LinearRegression(fit_intercept=False).fit(y[:, np.newaxis], x)
  y_on_x = LinearRegression(fit_intercept=False).fit(x[:, np.newaxis], y)

  # Coefficient values
  beta1 = y_on_x.coef_[0]
  alpha1 = x_on_y.coef_[0]
  print("beta1, alpha1")
  print(beta1,alpha1)
  # Verify inverse relationship
  print("Are beta and alpha reciprocally close?")
  print(np.isclose(beta1 * alpha1, 1))
  if (np.isclose(beta1 * alpha1, 1)):
    break
  denominator *= 10
  noise = noise / denominator

# %% [markdown]
# - *Reference: <https://stats.stackexchange.com/questions/22718/what-is-the-difference-between-linear-regression-on-y-with-x-and-x-with-y>*

# %% [markdown]
# ## (d) For the regression of Y onto X without an intercept, the t-statistic for $H_{0} : \beta = 0$ takes the form $\hat{\beta} / SE(\hat{\beta})$, where $\hat{\beta}$ is given by (3.38), and where

# %% [markdown]
# $$
# \Large SE(\hat{\beta}) = \sqrt{\frac {\sum_{i=1}^{n} \left (y_{i} - x_{i}\hat{\beta} \right )^{2}} {\left (n - 1 \right )\sum_{i^{'}=1}^n x_{i^{'}}^2}}
# $$
#

# %% [markdown]
# ## (These formulas are slightly different from those given in Sections 3.1.1 and 3.1.2, since here we are performing regression without an intercept.) Show algebraically, and confirm numerically in Python, that the t-statistic can be written as

# %% [markdown]
# $$
# \Large \frac {\left (n - 1 \right ) \sum_{i=1}^n x_{i}y_{i}} {\sqrt {\left (\sum_{i=1}^n x_{i}^2 \right ) \left (\sum_{i=1}^n y_{i}^2 \right ) - \left (\sum_{i^{'}=1}^{n}x_{i^{'}}y_{i^{'}} \right )^{2}}}
# $$

# %% [markdown]
# ![Solution](Exercise11.jpg "Solution")

# %%
df = generate_data()
formula = "y ~ x + 0"
model = smf.ols(f'{formula}', df)
results = model.fit()
result_df = get_results_df(results)
t = result_df["tstatistic"]

# %% [markdown]
# ### The calculated t-statistic from the formula above is:

# %%
x = df["x"]
y = df["y"]
numerator = np.sqrt(results.nobs - 1) * np.sum(x * y)
denominator = np.sqrt(np.sum(x**2) * np.sum(y**2) - np.sum(x * y) ** 2)
tstat = numerator / denominator

# %%
np.isclose(t, tstat)

# %% [markdown]
# ## (e) Using the results from (d), argue that the t-statistic for the regression of y onto x is the same as the t-statistic for the regression of x onto y.

# %% [markdown]
# - This is evident when we swap y and x in the derived formula that the result is the same.
# - This is also what we observed when we regressed y on x and x on y (without intercept)
# - Hence, the T-statistic is the same in both cases

# %% [markdown]
# ## (f) In Python, show that when regression is performed with an intercept, the t-statistic for $H_{0} : \beta_{1} = 0$ is the same for the regression of y onto x as it is for the regression of x onto y.

# %%
formula = "y ~ x"
model = smf.ols(f'{formula}', df)
results = model.fit()
result_df = get_results_df(results)
tx = result_df["tstatistic"].iloc[1]

# %%
formula = "x ~ y"
model = smf.ols(f'{formula}', df)
results = model.fit()
result_df = get_results_df(results)
ty = result_df["tstatistic"].iloc[1]

# %%
np.isclose(tx,ty)

# %%
allDone();
