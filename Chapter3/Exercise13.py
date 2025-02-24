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
# # Exercise 13

# %% [markdown]
# ## Import notebook funcs

# %%
from notebookfuncs import *

# %% [markdown]
# from IPython.display import Markdown, display, Math, Latex
#
# def printmd(string):
#     display(Markdown(string))

# %% [markdown]
# ## In this exercise you will create some simulated data and will fit simple linear regression models to it. Make sure to use the default random number generator with seed set to 1 prior to starting part (a) to ensure consistent results.

# %% [markdown]
# ### (a) Using the normal() method of your random number generator, create a vector, x, containing 100 observations drawn from a N (0, 1) distribution. This represents a feature, X.

# %%
import numpy as np
import pandas as pd
import markdown

RNG = np.random.default_rng(1)

def generate_data(mean=0.0, sd=1.0, N=100):
    series = RNG.normal(size=N, loc=mean, scale=sd)
    return series

x = generate_data(mean=0.0, sd=1.0, N = 100);

# %% [markdown]
# ### (b) Using the normal() method, create a vector, eps, containing 100 observations drawn from a N (0, 0.25) distribution—a normal distribution with mean zero and variance 0.25.

# %%
eps = generate_data(mean=0.0, sd = 0.25, N = 100);

# %% [markdown]
# ### (c) Using x and eps, generate a vector y according to the model 
# #### $Y = −1 + 0.5 * X + \epsilon$

# %%
beta_0 = -1
beta_1 = 0.5
y = -1 + 0.5 * x + eps;

# %% [markdown]
# #### What is the length of the vector y? What are the values of $\beta_0$ and $\beta_1$ in this linear model?

# %%
len(y)

# %%
display(Math(rf"\beta_0 = {beta_0} \: and \: \beta_1 = {beta_1}"))

# %% [markdown]
# ### (d) Create a scatterplot displaying the relationship between x and y. Comment on what you observe.

# %%
import seaborn as sns
df =pd.DataFrame({"x": x, "y":y})
sns.scatterplot(data=df, x="x", y="y");

# %% [markdown]
# - *There appears to be a positive linear relationship between x and y when viewed visually through the scatterplot.*

# %% [markdown]
# ### (e) Fit a least squares linear model to predict y using x. Comment on the model obtained. How do $\hat{\beta_0}$ and $\hat{\beta_1}$ compare to $\beta_0$ and $\beta_1$ ?

# %%
import statsmodels.formula.api as smf

def get_results_df(results):
    result_df = pd.DataFrame(
        {
          "coefficients": results.params,
            "coefficients-se": results.bse,
            "tstatistic": results.tvalues,
            "p-value": results.pvalues,
            "r-squared": results.rsquared,
            "r-squared-adjusted": results.rsquared_adj,
            "pearson_coefficient": np.sqrt(results.rsquared),
            "rss": results.ssr,
            "sd_residuals": np.sqrt(results.mse_resid)
        }
    )
    return result_df

def regress_y_on_x(df):
  formula = "y ~ x"
  model = smf.ols(f"{formula}", df)
  results = model.fit()
  result_df = get_results_df(results)
  return results, result_df

orig_res, result_df = regress_y_on_x(df)
result_df

# %%
printmd(r"The $\: \hat{\beta_0}$ = " + str(orig_res.params.iloc[0]) + r" and $\hat{\beta_1}$ = " + str(orig_res.params.iloc[1]) + r" compare quite favourably to the population parameters $\beta_0$ = " +  str(beta_0) + r" and $\beta_1$ = " + str(beta_1) +".")


# %% [markdown]
# ### (f) Display the least squares line on the scatterplot obtained in (d). Draw the population regression line on the plot, in a different color. Use the legend() method of the axes to create an appropriate legend.

# %%
def draw_regplot(df):
  ax = sns.regplot(x="x",y="y",data=df,label="estimate", color="blue");
  x = df["x"]
  y_original = -1 + 0.5 * x
  sns.regplot(x=x, y=y_original, scatter=False,label="population",color="red",ax=ax);
  ax.legend();

draw_regplot(df)

# %% [markdown]
# ### (g) Now fit a polynomial regression model that predicts $y$ using $x$ and $x^2$. Is there evidence that the quadratic term improves the model fit? Explain your answer.

# %%
simple_results = result_df
formula = "y ~ x + I(x**2)"
model = smf.ols(f"{formula}", df)
results = model.fit()
result_df = get_results_df(results)

# %%
np.isclose(simple_results["r-squared"].iloc[0], result_df["r-squared"].iloc[0])

# %%
np.isclose(simple_results["r-squared-adjusted"].iloc[0], result_df["r-squared-adjusted"].iloc[0])

# %%
ci = results.conf_int(alpha=0.05) 
ci[(ci[0] < 0) & (ci[1] > 0)]

# %% [markdown]
# - The $R^2$ does not change significantly with the polynomial regression. So there is no evidence that the quadratic term improves the model fit.
# - The $R^2 adjusted$ does increase but the coefficient for the polynomial term is not significant.
# - Additionally, the confidence interval for the polynomial term spans both -ve and +ve axes i.e., zero lies in the range of the confidence interval.

# %% [markdown]
# ### (h) Repeat (a)–(f) after modifying the data generation process in such a way that there is less noise in the data. The model (3.39) should remain the same. You can do this by decreasing the variance of the normal distribution used to generate the error term $\epsilon$ in (b). Describe your results.

# %% [markdown]
# #### We decrease the standard deviation of the error terms or noise to 0.05 from 0.25.

# %%
eps = generate_data(mean=0.0, sd = 0.05, N = 100);

# %%
y = -1 + 0.5 * x + eps;
df = pd.DataFrame({"x": x, "y": y});

# %%
less_noisier_results, result_df = regress_y_on_x(df)
result_df

# %%
draw_regplot(df)

# %% [markdown]
# - We conclude that the less noiser the data, the closer the fit of the regression to the population parameters.

# %% [markdown]
# ### (i) Repeat (a)–(f) after modifying the data generation process in such a way that there is more noise in the data. The model (3.39) should remain the same. You can do this by increasing the variance of the normal distribution used to generate the error term $\epsilon$ in (b). Describe your results.

# %% [markdown]
# #### We increase the standard deviation of the error terms or noise to 1 from 0.25.

# %%
eps = generate_data(mean=0.0, sd = 1, N = 100);

# %%
y = -1 + 0.5 * x + eps;

# %%
df = pd.DataFrame({"x": x, "y": y});

# %%
noisier_results, result_df = regress_y_on_x(df)
result_df

# %%
draw_regplot(df)

# %% [markdown]
# - We conclude that the noisier the data, the more the drift of the fit from the population paramaters especially when it comes to estimating the slope of the fitted line.

# %% [markdown]
# ### (j) What are the confidence intervals for $\beta_0$ and $\beta_1$ based on the original data set, the noisier data set, and the less noisy data set? Comment on your results.

# %%
print("Original dataset")
print(orig_res.conf_int(alpha=0.05))
print()
print("Less noisy dataset")
print(less_noisier_results.conf_int(alpha=0.05))
print()
print("More noisy dataset")
print(noisier_results.conf_int(alpha=0.05))

# %%
print("Standard errors of coefficients")
print()
print("Original dataset")
print(orig_res.bse)
print()
print("Less noisy dataset")
print(less_noisier_results.bse)
print()
print("More noisy dataset")
print(noisier_results.bse)

# %% [markdown]
# - We can conclude that the noisier the dataset, the more likely that the confidence intervals are wider so that the population parameters actually reside witin its range.
# - This is because the Standard errors of the parameters are wider when the data is noisier.

# %%
allDone();
