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
    results_df = get_results_df(results)
    results_df = results_df[results_df.index != "Intercept"]
    rows = pd.concat([rows,results_df])

  return rows


regress_for_each_predictor(data=Boston,response="crim")

# %% [markdown]
# ### Describe your results.

# %% [markdown]
#
# ### In which of the models is there a statistically significant association between the predictor and the response?

# %% [markdown]
# ### Create some plots to back up your assertions.

# %% [markdown]
# ## (b) Fit a multiple regression model to predict the response using all of the predictors.

# %% [markdown]
# ## Describe your results.

# %% [markdown]
# ## For which predictors can we reject the null hypothesis $H_0 : \beta_j = 0$?

# %% [markdown]
# ## (c) How do your results from (a) compare to your results from (b)?

# %% [markdown]
# ### Create a plot displaying the univariate regression coefficients from (a) on the x-axis, and the multiple regression coefficients from (b) on the y-axis. That is, each predictor is displayed as a single point in the plot. Its coefficient in a simple linear regression model is shown on the x-axis, and its coefficient estimate in the multiple linear regression model is shown on the y-axis.

# %% [markdown]
# ## (d) Is there evidence of non-linear association between any of the predictors and the response?

# %% [markdown]
# ### To answer this question, for each predictor X, fit a model of the form $Y = \beta_0 + \beta_1 * X + \beta_2 * X^2 + \beta_3 * X^3 + \epsilon$

# %%
allDone();
