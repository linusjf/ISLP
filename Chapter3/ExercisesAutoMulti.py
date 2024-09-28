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
# # Multilinear Regression: Auto dataset

# %% [markdown]
# ## Import notebook functions

# %%
from notebookfuncs import *

# %% [markdown]
# ## Import standard libraries

# %%
import numpy as np
import pandas as pd

pd.set_option('display.max_rows', 1000)
pd.set_option('display.max_columns', 1000)
pd.set_option('display.width', 1000)
pd.set_option("display.max.colwidth", None)
import matplotlib.pyplot as plt
from matplotlib.pyplot import subplots
import seaborn as sns
import itertools

# %% [markdown]
# ## New imports

# %%
import statsmodels.api as sm

# %% [markdown]
# ## Import statsmodels.objects

# %%
from statsmodels.stats.outliers_influence import variance_inflation_factor as VIF
from statsmodels.stats.outliers_influence import summary_table
from statsmodels.stats.anova import anova_lm
import statsmodels.formula.api as smf

# %% [markdown]
# ## Import ISLP objects

# %%
import ISLP
from ISLP import models
from ISLP import load_data
from ISLP.models import (ModelSpec as MS, summarize, poly)

# %% [markdown]
# #### Import user functions

# %%
from userfuncs import *

# %% [markdown]
# #### Set level of significance (alpha)

# %%
LOS_Alpha = 0.01

# %%
Auto = load_data('Auto')
Auto = Auto.sort_values(by=['year'], ascending=True)
Auto.head()
Auto.columns

# %%
Auto.shape

# %%
Auto.describe()

# %% [markdown]
# ## 9. This question involves the use of multiple linear regression on the Auto data set.

# %% [markdown]
# ### (a) Produce a scatterplot matrix which includes all of the variables in the data set.

# %%
pd.plotting.scatter_matrix(Auto, figsize=(14, 14));

# %% [markdown]
# ### (b) Compute the matrix of correlations between the variables using the DataFrame.corr() method.

# %%
Auto.corr()

# %% [markdown]
# ### (c) Use the sm.OLS() function to perform a multiple linear regression with mpg as the response and all other variables except name as the predictors. Use the summarize() function to print the results. Comment on the output. For instance:

# %% [markdown]
# ## Convert year and origin columns to categorical types

# %%
Auto["origin"] = Auto["origin"].astype("category")
Auto['origin'] = Auto['origin'].cat.rename_categories({
    1: 'America',
    2: 'Europe',
    3: 'Japan'
})
Auto["year"] = Auto["year"].astype("category")
Auto.describe()

# %%
sns.relplot(Auto,
            x="year",
            y="weight",
            col="origin",
            hue="cylinders",
            style="cylinders",
            estimator='mean',
            kind="line");

# %% [markdown]
# #### The weight of the 8-cylinder American made models show a decline from the highs of 1972. It can also be seen that American made cars are heavier than their European and Japanese counterparts especially in the most common models with 4 cylinders.

# %%
sns.relplot(Auto,
            x="year",
            y="mpg",
            col="origin",
            hue="cylinders",
            style="cylinders",
            estimator='mean',
            kind="line");

# %% [markdown]
# #### It can be seen that after the [oil shock of 1973](https://en.wikipedia.org/wiki/1973_oil_crisis) and the regulations and actions taken by the US government, the mileage for American made cars rose across all models. This was, however, matched by the European and Japanese models which were already lighter and more fuel efficient.

# %% [markdown]
# ### Encode categorical variables as dummy variables dropping the first to remove multicollinearity.


# %%
def categorize_for_oil_shock(row):
    # we add 3 years because it takes approximately that long for car manufacturers to introduce a new model
    if row["year"] in (70, 71, 72, 73, 74, 75, 76):
        return 0
    return 1


Auto["oilshock"] = Auto.apply(categorize_for_oil_shock, axis=1)

# %%
Auto.boxplot(column="mpg", by=["oilshock", "origin"]);

# %%
Auto_os = Auto.drop(["year"], axis=1)
Auto_os.columns

# %%
# standardizing dataframes
Auto_os["oilshock"] = Auto_os["oilshock"].astype("category")
Auto_os = Auto_os.apply(standardize)
Auto_os.describe()

# %%
Auto_os = pd.get_dummies(Auto_os,
                         columns=list(["origin"]),
                         drop_first=True,
                         dtype=np.uint8)
Auto_os.columns

# %%
y = Auto_os["mpg"];

# %%
cols = list(Auto_os.columns)
cols.remove("mpg")
formula = ' + '.join(cols)
model = smf.ols(f'mpg ~ {formula}', data=Auto_os)
results = model.fit()
results.summary()

# %% [markdown]
# #### i. Is there a relationship between the predictors and the response? Use the anova_lm() function from statsmodels to answer this question.
# #### ii. Which predictors appear to have a statistically significant relationship to the response?
# #### iii. What does the coefficient for the year variable suggest?

# %%
anova_lm(results)

# %% [markdown]
# #### There seems to be a statistical relationship between all of the predictors and the response variable, mpg, except for acceleration.
# #### Even though some of the categorical variables are insignificant, even if one of the levels is significant, it is advisable to retain them all in the model.
#
# <https://stats.stackexchange.com/questions/24298/can-i-ignore-coefficients-for-non-significant-levels-of-factors-in-a-linear-mode>
#
# Note: Year has been converted to a categorical variable oilshock to better capture the effects of the oil shock of 1973 on the mileage.

# %% [markdown]
# ### (d) Produce some of diagnostic plots of the linear regression fit as described in the lab. Comment on any problems you see with the fit. Do the residual plots suggest any unusually large outliers? Does the leverage plot identify any observations with unusually high leverage?

# %% [markdown]
# #### Before producing the diagnostic plots, let's first test for collinearity using correlation matrix and variance inflation factors.

# %%
Auto_os.corr(numeric_only=True)

# %%
vifdf = calculate_VIFs("mpg ~ " + " + ".join(Auto_os.columns) + " - mpg",
                       Auto_os)
vifdf

# %%
identify_highest_VIF_feature(vifdf)

# %%
vifdf = calculate_VIFs(
    "mpg ~ " + " + ".join(Auto_os.columns) + " - mpg - displacement", Auto_os)
vifdf

# %%
identify_highest_VIF_feature(vifdf)

# %% [markdown]
# ### Linear Regression for mpg ~ cylinders + horsepower + weight + acceleration + oilshock + origin_Europe + origin_Japan

# %%
cols = list(Auto_os.columns)
cols.remove("mpg")
cols.remove("displacement")
formula = ' + '.join(cols)
results = perform_analysis("mpg", formula, Auto_os);

# %%
identify_least_significant_feature(results, alpha=LOS_Alpha)

# %% [markdown]
# #### Linear Regression after dropping acceleration. The model now is mpg ~ cylinders + horsepower + weight + oilshock +  origin_Europe + origin_Japan

# %%
cols.remove("acceleration")
formula = ' + '.join(cols)
results = perform_analysis("mpg", formula, Auto_os)
simple_model = results
models = []
models.append({
    "name": "simple_model",
    "model": results.model.formula,
    "R-squared adjusted": results.rsquared_adj
})

# %% [markdown]
# #### Linear Regression after dropping cylinders. The model now is mpg ~  horsepower + weight + oilshock +  origin_Europe + origin_Japan

# %%
cols.remove("cylinders")
formula = ' + '.join(cols)
results = perform_analysis("mpg", formula, Auto_os)
simple_model = results
models = []
models.append({
    "name": "simple_model",
    "model": results.model.formula,
    "R-squared adjusted": results.rsquared_adj
})

# %% [markdown]
# #### We can now try and plot the diagnostics for the model.

# %%
TSS = np.sum((y - np.mean(y))**2)
TSS
RSS = np.sum((y - results.fittedvalues)**2)
RSS
RSE = np.sqrt(RSS / results.df_model)
display("RSE " + str(RSE))
display("R-squared adjusted : " + str(results.rsquared_adj))
display("F-statistic : " + str(results.fvalue))

# %%
display_residuals_plot(results)

# %% [markdown]
# There is some evidence of non-linearity and heteroskedasticity from the residuals plot above.

# %% [markdown]
# ### (e) Fit some models with interactions as described in the lab. Do any interactions appear to be statistically significant?

# %%
formula = ' + '.join(cols)
formula += " + " + "horsepower: weight"
results = perform_analysis("mpg", formula, Auto_os)
numeric_interactions = results
models.append({
    "name": "numeric_interactions",
    "model": results.model.formula,
    "R-squared adjusted": results.rsquared_adj
})

# %%
formula = ' + '.join(cols)
formula += " + " + "horsepower: weight"
formula += " + " + "oilshock: weight"
formula += " + " + "oilshock: horsepower"
results = perform_analysis("mpg", formula, Auto_os)
oilshock_interactions = results
models.append({
    "name": "oilshock_interactions",
    "model": results.model.formula,
    "R-squared adjusted": results.rsquared_adj
})

# %%
formula = ' + '.join(cols)
formula += " + " + "oilshock: horsepower"
formula += " + " + "origin_Europe: horsepower"
formula += " + " + "origin_Japan: horsepower"
formula += " + " + "origin_Europe: weight"
formula += " + " + "origin_Japan: weight"
formula += " + " + "oilshock: weight"
formula += " + " + "oilshock: horsepower"
results = perform_analysis("mpg", formula, Auto_os);
origin_interactions = results;

# %% [markdown]
# + From the above analysis, we can see that there is no significant interaction between origin and weight.
# + So we can omit them from the model.

# %%
formula = ' + '.join(cols)
formula += " + " + "oilshock: horsepower"
formula += " + " + "origin_Europe: horsepower"
formula += " + " + "origin_Japan: horsepower"
formula += " + " + "oilshock: weight"
formula += " + " + "oilshock: horsepower"
results = perform_analysis("mpg", formula, Auto_os);
origin_interactions = results;

# %% [markdown]
# + From the above analysis, it is evident that with the interaction between origin and horsepower, the interaction between oilshock and weight and horsepower is insignificant. We can drop these from the model as well.

# %%
formula = ' + '.join(cols)
formula += " + " + "oilshock: horsepower"
formula += " + " + "origin_Europe: horsepower"
formula += " + " + "origin_Japan: horsepower"
results = perform_analysis("mpg", formula, Auto_os);
origin_interactions = results
models.append({
    "name": "origin_interactions",
    "model": results.model.formula,
    "R-squared adjusted": results.rsquared_adj
})

# %%
display_residuals_plot(results)

# %%
anova_lm(simple_model, numeric_interactions, oilshock_interactions,
         origin_interactions)

# %%
pd.DataFrame(models)

# %% [markdown]
# ### (f) Try a few  different transformations of the variables, such as log(X), √X, X<sup>2</sup> . Comment on your findings.

# %%
formula = simple_model.model.formula
formula = formula[formula.rindex("~") + 1:]
# Add higher order transformations for horsepower and weight
formula += " + " + "I(horsepower**2)"
formula += " + " + "I(weight**2)"
results = perform_analysis("mpg", formula, Auto_os);
squared_transformations = results
models.append({
    "name": "squared_transformation",
    "model": results.model.formula,
    "R-squared adjusted": results.rsquared_adj
})

# %%
display_residuals_plot(results)

# %%
anova_lm(simple_model, squared_transformations)

# %%
pd.DataFrame(models)

# %% [markdown]
# - Since we've standardized the variables, we cannot run log or square root transformations on the negative valued columns.

# %% [markdown]
# - We can reload the data and run the log and sqrt transformations on the original un-standardized data.

# %%
Auto = load_data('Auto')
Auto = Auto.sort_values(by=['year'], ascending=True)
Auto.columns

# %%
print("Minimums:")
print(Auto.min())
print("Maximums:")
print(Auto.max())

# %% [markdown]
# - From the above, we can see that the values for displacement, horsepower and weight are quite large.
# - Hence, we log or square root transform only these variables.

# %% [markdown]
# ### Now let's categorize the variables 

# %%
Auto["origin"] = Auto["origin"].astype("category")
Auto['origin'] = Auto['origin'].cat.rename_categories({
    1: 'America',
    2: 'Europe',
    3: 'Japan'
})
Auto["year"] = Auto["year"].astype("category")
Auto["oilshock"] = Auto.apply(categorize_for_oil_shock, axis=1)

# %% [markdown]
# ## Log Transformed Model

# %%
Auto_log = Auto.copy(deep=True);

# %%
Auto_log["log_displacement"] = np.log(Auto_log["displacement"])
Auto_log["log_horsepower"] = np.log(Auto_log["horsepower"])
Auto_log["log_weight"] = np.log(Auto_log["weight"])
Auto_log = Auto_log.drop(columns=["displacement", "weight", "horsepower", "year",]);
Auto_log.columns

# %%
Auto_log.corr(numeric_only=True)

# %%
Auto_log = pd.get_dummies(Auto_log,
                         columns=list(["origin"]),
                         drop_first=True,
                         dtype=np.uint8)
Auto_log.columns

# %%
cols = list(Auto_log.columns)
cols.remove("mpg")

# %%
vifdf = calculate_VIFs("mpg ~ " + " + ".join(cols),
                       Auto_log)
vifdf

# %%
identify_highest_VIF_feature(vifdf)

# %%
cols.remove("log_displacement")
vifdf = calculate_VIFs("mpg ~ " + " + ".join(cols),
                       Auto_log)
vifdf

# %%
identify_highest_VIF_feature(vifdf)

# %%
cols.remove("log_horsepower")
vifdf = calculate_VIFs("mpg ~ " + " + ".join(cols),
                       Auto_log)
vifdf

# %%
identify_highest_VIF_feature(vifdf)

# %%
formula = ' + '.join(cols)
results = perform_analysis("mpg", formula,Auto_log);

# %%
identify_least_significant_feature(results, alpha=LOS_Alpha)

# %%
cols.remove("cylinders")
formula = ' + '.join(cols)
results = perform_analysis("mpg", formula,Auto_log);

# %%
identify_least_significant_feature(results, alpha=LOS_Alpha)

# %%
models.append({
    "name": "log_transformation",
    "model": results.model.formula,
    "R-squared adjusted": results.rsquared_adj
})

# %%
pd.DataFrame(models)

# %% [markdown]
# ## Square Root Transformed Model

# %%
Auto_sqrt = Auto.copy(deep=True);

# %%
Auto_sqrt["sqrt_displacement"] = np.sqrt(Auto_sqrt["displacement"])
Auto_sqrt["sqrt_horsepower"] = np.sqrt(Auto_sqrt["horsepower"])
Auto_sqrt["sqrt_weight"] = np.sqrt(Auto_sqrt["weight"])
Auto_sqrt = Auto_sqrt.drop(columns=["displacement", "weight", "horsepower", "year",]);
Auto_sqrt.columns

# %%
Auto_sqrt.corr(numeric_only=True)

# %%
Auto_sqrt = pd.get_dummies(Auto_sqrt,
                         columns=list(["origin"]),
                         drop_first=True,
                         dtype=np.uint8)
Auto_sqrt.columns

# %%
cols = list(Auto_sqrt.columns)
cols.remove("mpg")

# %%
vifdf = calculate_VIFs("mpg ~ " + " + ".join(cols),
                       Auto_sqrt)
vifdf

# %%
identify_highest_VIF_feature(vifdf)

# %%
cols.remove("sqrt_displacement")
vifdf = calculate_VIFs("mpg ~ " + " + ".join(cols),
                       Auto_sqrt)
vifdf

# %%
identify_highest_VIF_feature(vifdf)

# %%
cols.remove("sqrt_horsepower")
vifdf = calculate_VIFs("mpg ~ " + " + ".join(cols),
                       Auto_sqrt)
vifdf

# %%
identify_highest_VIF_feature(vifdf)

# %%
formula = ' + '.join(cols)
results = perform_analysis("mpg", formula,Auto_sqrt);

# %%
identify_least_significant_feature(results, alpha=LOS_Alpha)

# %%
cols.remove("cylinders")
formula = ' + '.join(cols)
results = perform_analysis("mpg", formula,Auto_sqrt);

# %%
identify_least_significant_feature(results, alpha=LOS_Alpha)

# %%
cols.remove("acceleration")
formula = ' + '.join(cols)
results = perform_analysis("mpg", formula,Auto_sqrt);

# %%
models.append({
    "name": "sqrt_transformation",
    "model": results.model.formula,
    "R-squared adjusted": results.rsquared_adj
})

# %%
pd.DataFrame(models)

# %%
allDone()
