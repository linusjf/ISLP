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

# %%
from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"

## Import up sound alert dependencies
from IPython.display import Audio, display

def allDone():
  url = "https://sound.peal.io/ps/audios/000/064/733/original/youtube_64733.mp3"
  display(Audio(url=url, autoplay=True))


# %% [markdown]
# # Applied : Auto dataset - Simple Linear Regression

# %% [markdown]
# ## Import standard libraries

# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.pyplot import subplots

# %% [markdown]
# ## New imports

# %%
import statsmodels.api as sm

# %% [markdown]
# ## Import statsmodel.objects

# %%
from statsmodels.stats.outliers_influence import variance_inflation_factor as VIF
from statsmodels.stats.outliers_influence import summary_table
from statsmodels.stats.anova import anova_lm

# %% [markdown]
# ## Import ISLP objects

# %%
import ISLP
from ISLP import models
from ISLP import load_data
from ISLP.models import (ModelSpec as MS, summarize, poly)

# %%
Auto = load_data('Auto')
Auto.columns

# %%
Auto.shape

# %%
Auto.describe()

# %% [markdown]
# ## Convert cylinders and origin columns to categorical types

# %%
Auto["cylinders"] = Auto["cylinders"].astype("category")
Auto["origin"] = Auto["origin"].astype("category")
Auto.describe()

# %% [markdown]
# ## 8) This question involves the use of Simple Linear Regression on the Auto dataset

# %% [markdown]
# ### (a) Use the sm.OLS() function to perform a simple linear regression with mpg as the response and horsepower as the predictor.
# ### Use the summarize() function to print the results.
# ### Comment on the output. For example:
#
# #### i. Is there a relationship between the predictor and the response?
#
# #### ii. How strong is the relationship between the predictor and the response?
#
# #### iii. Is the relationship between the predictor and the response positive or negative?
#
# #### iv. What is the predicted mpg associated with a horsepower of 98? What are the associated 95 % confidence and prediction intervals?

# %%
y = Auto["mpg"]
y.head()

# %%
design = MS(["horsepower"])
design = design.fit(Auto)
X = design.transform(Auto)

# %%
model = sm.OLS(y, X)
results = model.fit()
summarize(results)

# %% [raw]
# - There is evidence of a linear relationship between horespower and the response mpg.

# %%
results.summary()

# %% [markdown]
# + The R<sup>2</sup> value of 60.6% indicates that the regression of horsepower on mpg explains 60.6% of the variation in the model.

# %% [markdown]
# + The relationship between horsepower and  mpg is negative, i.e., an increase in hp of 1 unit decreases the mileage by 0.1578 miles. An increase in the car's output in power is offset by a decrease in its economy.

# %%
design = MS(["horsepower"])
new_df = pd.DataFrame({"horsepower": [98]})
design = design.fit(new_df)
newX = design.transform(new_df)

# %%
new_predictions = results.get_prediction(newX)
mileage = new_predictions.predicted_mean[0]
mileage

# %% [markdown]
# - The predicted mileage for a horsepower of 98 is 24.47 mpg.

# %%
new_predictions.conf_int(alpha=0.05)

# %% [markdown]
# + The 95% confidence interval is (23.97, 24.96)

# %%
new_predictions.conf_int(alpha=0.05, obs=True)

# %% [markdown]
# + The 95% prediction interval is (14.82, 34.13)

# %% [markdown]
# ### (b) Plot the response and the predictor in a new set of axes ax.
# ### Use the ax.axline() method or the abline() function defined in the lab to display the least squares regression line.

# %%
ax = Auto.plot.scatter("horsepower", "mpg");
ax.axline((ax.get_xlim()[0], results.params.iloc[0]), slope=results.params.iloc[1], color="r", linewidth=3);

# %% [markdown]
# + The least squares regression line is plotted above using ax.axline(). The plot displays some evidence of non-linearity in the relationship between horsepower and mpg.

# %% [markdown]
# ### (c) Produce some of diagnostic plots of the least squares regression fit as described in the lab.
# ### Comment on any problems you see with the fit.

# %%
#### Plot of fitted values versus residuals.

# %%
_, ax = subplots(figsize=(8,8))
ax.scatter(results.fittedvalues, results.resid)
ax.set_xlabel("Fitted values")
ax.set_ylabel("Residuals")
ax.axhline(0, c='k', ls='--');

# %% [markdown]
# #### We can also plot the residuals vs predictor plot where horsepower is the predictor.

# %%
_, ax = subplots(figsize=(8,8))
ax.scatter(Auto["horsepower"], results.resid)
ax.set_xlabel("Horsepower")
ax.set_ylabel("Residuals")
ax.axhline(0, c='k', ls='--');

# %% [markdown]
# #### Conclusions:
# + There is evidence of non-linearity in the relationship between residuals and fitted values.
# + There is evidence of heteroskedasticity i.e., non-constant variance in the residuals across the fitted values.

# %%
RSS = np.sum((y - results.fittedvalues) ** 2)
RSS

# %%
RSE = np.sqrt(RSS/ (Auto.shape[0] - 2))
RSE

# %% [markdown]
# ### OLSResults.scale()
#
# + Gives us a scale factor for the covariance matrix.
# + The Default value is ssr/(n-p). Note that the square root of scale is often called the standard error of the regression.
#
# + <https://www.statsmodels.org/dev/generated/statsmodels.regression.linear_model.OLSResults.scale.html>

# %%
np.sqrt(results.scale)

# %%
mpg_mean = Auto["mpg"].mean()

# %%
print("Percentage error in mpg estimation using model above is: " )
np.round(RSE / mpg_mean * 100, decimals = 2)

# %% [markdown]
# ### Leverage statistics

# %%
infl = results.get_influence()
_, ax = subplots(figsize=(8,8))
ax.scatter(np.arange(X.shape[0]),infl.hat_matrix_diag)
ax.set_xlabel("Index")
ax.set_ylabel("Leverage")
high_leverage = np.argmax(infl.hat_matrix_diag)
max_leverage = np.max(infl.hat_matrix_diag)
print("Max leverage point:")
print(high_leverage, np.round(max_leverage, decimals = 2) )
ax.plot(high_leverage, max_leverage, "ro");

# %% [markdown]
# #### Outlier identification using Standardized Residuals versus Fitted Values plot

# %%
_, ax = subplots(figsize=(8,8))
ax.scatter(results.fittedvalues, results.resid_pearson)
ax.set_xlabel("Fitted values for mpg")
ax.set_ylabel("Standardized residuals")
ax.axhline(0, c='k', ls='--');
outliers_indexes = np.where((results.resid_pearson > 3.0) | (results.resid_pearson < -3.0))[0]
for idx in range(len(outliers_indexes)):
  ax.plot(results.fittedvalues.iloc[outliers_indexes[idx]],
          results.resid_pearson[outliers_indexes[idx]], "ro");
print("Outlier rows: ")
print(Auto.iloc[outliers_indexes])

# %% [markdown]
# Conclusions:
# + From the standardized residuals versus fitted values, there are two outliers present in the data.
# + These points can be investigated further whether to retain them in the dataset.
#
# Note:
# - We could drop the outliers from the data and regress the model without these points. That is an exercise for you!

# %% [markdown]
# #### We can also plot residuals versus order.

# %%
_, ax = subplots(figsize=(14,8))
ax.scatter(np.arange(X.shape[0]),results.resid)
ax.set_xlabel("Observation Order")
ax.set_ylabel("Residuals");
ax.axhline(0, c='k', ls='--');

# %% [markdown]
# Conclusions:
# - While there seems to be little evidence of negative or positive correlation over time, there is evidence of underestimation from observations 300 onwards. There also seems to be a time trend in the data from observation 300 or so where the expectation of the model is that mpg will be lower, but the actual values are much higher. This indicates that fuel mileage improved much more than expected in the later models from observation 300 onwards. This indicates that column year should be added to the model.

# %%
sm.qqplot(results.resid,line="s");

# %%
# Plot histogram of residuals
plt.hist(results.resid, bins=10);

# %% [markdown]
# Conclusions:
# - From the above two plots for qq and histograms for residuals, we can deduce that the residuals are approximately normal.

# %% [markdown]
# References:
# <https://github.com/linusjf/LearnR/tree/development/Stats462>

# %%
allDone()
