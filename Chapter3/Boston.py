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
# # Exercise 15: This problem involves the Boston data set, which we saw in the lab for this chapter.

# %% [markdown]
# ## Import notebook funcs

# %%
from notebookfuncs import *

# %% [markdown]
# ## Import userfuncs

# %%
from userfuncs import *

# %% [markdown]
# ## Import libraries

# %%
from ISLP import load_data
from summarytools import dfSummary
import seaborn as sns

# %%
Boston = load_data("Boston")
dfSummary(Boston)


# %% [markdown]
# ## We will now try to predict per capita crime rate using the other variables in this data set.

# %% [markdown]
# ## In other words, per capita crime rate (crim) is the response, and the other variables are the predictors.

# %% [markdown]
# ## (a) For each predictor, fit a simple linear regression model to predict the response.

# %%
def regress_for_each_predictor(data=None,response=None):
  if (data is None or response is None):
    return None
  col_names = list(data.columns.values)
  col_names.remove(response)
  rows = None
  for col in col_names:
    formula = f"{response} ~ {col}"
    model = smf.ols(formula, data=data)
    results = model.fit()
    results_df = pd.DataFrame({"Regressor": col,
                              "Coefficient": results.params.iloc[1],
                              "P-value": results.pvalues.iloc[1],
                             "R-Squared": results.rsquared}, index=[0])
    rows = pd.concat([rows,results_df])

  rows.set_index(["Regressor"],inplace=True)
  rows.sort_values("R-Squared", inplace=True, ascending=False)
  return rows

regressors = regress_for_each_predictor(data=Boston,response="crim")

# %% [markdown]
# ### Describe your results.

# %%
regressors[regressors["P-value"] > 0.05]

# %% [markdown]
# - From the above, we see that $chas$ (Charles River dummy variable whether tract bounds river or not) is the only regressor that is not statistically significant in the simple linear regressions of each variable for $crim$.

# %% [markdown]
#
# ### In which of the models is there a statistically significant association between the predictor and the response?

# %% [markdown]
# - All of the variables except $chas$ are statistically significant in a regression for $crim$.

# %% [markdown]
# ### Create some plots to back up your assertions.

# %%
col_names = list(Boston.columns.values)
col_names.remove("crim")
sns.pairplot(Boston, x_vars=col_names,y_vars="crim");


# %% [markdown]
# ## (b) Fit a multiple regression model to predict the response using all of the predictors.

# %%
def regress_on_all_predictors(data=None,response=None):
  if (data is None or response is None):
    return None
  columns = list(Boston.columns.values)
  columns.remove(response)
  formula = response + " ~ " + " + ".join(columns)
  model = smf.ols(formula, data=data)
  results = model.fit()
  results_df = pd.DataFrame({"Regressor": columns,
                              "Coefficient": results.params[1:],
                              "P-value": results.pvalues[1:],
                             "R-Squared": results.rsquared})
  results_df.set_index(["Regressor"],inplace=True)
  return results_df

all_regressors = regress_on_all_predictors(data=Boston, response="crim")

# %% [markdown]
# ## Describe your results.

# %%
all_regressors[all_regressors["P-value"] > 0.05]

# %% [markdown]
# ## For which predictors can we reject the null hypothesis $H_0 : \beta_j = 0$?

# %%
all_regressors[all_regressors["P-value"] < 0.05]

# %% [markdown]
# ## (c) How do your results from (a) compare to your results from (b)?

# %% [markdown]
# - In the multilinear regression model over all regressors, we discover that only $zn$, $dis$, $rad$ and $medv$ are statistically significant and the rest aren't.
# - This implies that there is multicollinearity in the data.

# %% [markdown]
# ### Create a plot displaying the univariate regression coefficients from (a) on the x-axis, and the multiple regression coefficients from (b) on the y-axis. That is, each predictor is displayed as a single point in the plot. Its coefficient in a simple linear regression model is shown on the x-axis, and its coefficient estimate in the multiple linear regression model is shown on the y-axis.

# %%
# merge the two dataframes
combined = regressors.merge(all_regressors, left_index=True, right_index=True, suffixes=["_simple", "_all"])
# select only the needed columns
combined = combined[["Coefficient_simple", "Coefficient_all"]]

# %%
import plotly.express as px
fig = px.scatter(combined, x="Coefficient_simple", y="Coefficient_all",color=combined.index)
fig.show()

# %% [markdown]
# ## (d) Is there evidence of non-linear association between any of the predictors and the response?

# %% [markdown]
# ### To answer this question, for each predictor X, fit a model of the form $Y = \beta_0 + \beta_1 * X + \beta_2 * X^2 + \beta_3 * X^3 + \epsilon$

# %%
allDone();
