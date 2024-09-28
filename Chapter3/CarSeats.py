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
# # Multilinear Regression: CarSeats dataset

# %% [markdown]
# ## Import notebook funcs

# %%
from notebookfuncs import *

# %% [markdown]
# ## Import ISLP objects

# %%
from ISLP import load_data

# %% [markdown]
# ## Import User Funactions

# %%
from userfuncs import *

# %% [markdown]
# ## Load dataset

# %%
Carseats = load_data('Carseats')
Carseats.head()

# %%
Carseats.shape

# %%
Carseats = Carseats.dropna()
Carseats.shape

# %% [markdown]
# ## Display dataset stats

# %%
Carseats.describe()

# %% [markdown]
# ## Set categorical types

# %%
Carseats["US"] = Carseats["US"].astype("category")
Carseats["Urban"] = Carseats["Urban"].astype("category")

# %% [markdown]
# ### (a) Fit a multiple regression model to predict Sales using Price, Urban, and US.

# %%
formula = "Price + Urban + US"
perform_analysis("Sales", formula, Carseats);

# %% [markdown]
# ### (b) Provide an interpretation of each coefficient in the model. Be careful—some of the variables in the model are qualitative!

# %% [markdown]
# - The coefficient of -0.0219 for Urban (True) indicates that the Sales are lesser by 22 units for an urban store as compared to a rural one. However, the p-value of 0.936 indicates that this difference is not significant and can be discounted or discarded.
# - The coefficent of 1.2006 for US (True) indicates that Sales are greater by 1201 units as compared to a non-US store.
# - The coefficient of -0.0545 for Price indicates that Sales decreases by 55 units per unit increase in cost all other things remaining constant.

# %% [markdown]
# ### (c) Write out the model in equation form, being careful to handle the qualitative variables properly.

# %% [markdown]
# - The equation can be written out as follows:
# - Sales (000s) = -0.0219 * Urban + 1.2006 * US -0.0545 * Price + 13.0435

# %% [markdown]
# ### (d) For which of the predictors can you reject the null hypothesis H0 : βj = 0?

# %% [markdown]
# - The p-value for the Urban predictor is 0.936 which is much higher than our chosen level of significance 0.01. So we cannot reject the null Hypothesis in this case that its coefficient is zero.
# - The p-values for US, Price and Intercept are zero. Hence, we reject the null hypothesis for them that their coefficients are zero.

# %% [markdown]
# ### (e) On the basis of your response to the previous question, fit a smaller model that only uses the predictors for which there is evidence of association with the outcome.

# %%
formula = "Price + US"
results = perform_analysis("Sales", formula, Carseats);

# %% [markdown]
# ### (f) How well do the models in (a) and (e) fit the data?

# %% [markdown]
# + Model(a) has an explanatory value R<sup>2</sup> adjusted value of 0.234
# + Model(e) has an explanatory value R<sup>2</sup> adjusted value of 0.235

# %% [markdown]
# ### (g) Using the model from (e), obtain 95 % confidence intervals for the coefficient(s).

# %% [markdown]
# From the summary analysis, it can be seen that the 95% confidence limits for the three terms are as follows:
# + Intercept (11.790, 14.271)
# + US[T.Yes] (0.692, 1.708)
# + Price (-0.065, -0.044)
# + None of them include zero in their range unlike that for Urban[T.Yes] in Model(a) which is another indicator that the coefficient is not significant.

# %% [markdown]
# ### (h) Is there evidence of outliers or high leverage observations in the model from (e)?

# %% [markdown]
# #### We can check for presence of outliers by plotting the residuals plot and seeing if there are any outliers.

# %%
display_residuals_plot(results)

# %% [markdown]
# - From the plot above, there doesn't appear to be any obvious outliers.

# %% [markdown]
# #### We can plot studentized residuals to see whether there are any visible there.

# %%
display_studentized_residuals(results)

# %% [markdown]
# - From the above plot, no observation lies outside the (-3,3) range. Hence, we can safely conclude that there are no evident outliers in the dataset.

# %%
display_hat_leverage_cutoffs(results)

# %% [markdown]
# - We can see from the above graph that we have a few leverage points that exceed the cutoff of 3 * average leverage value. These are plotted in red.
# - The ones in yellow exceed the less conservative estimate of 2 * average leverage value
# - We could also use more conservative estimates for the cutoff of either 4 * average leverage value or 5 * average cutoff value

# %% [markdown]
# References:
# - <https://online.stat.psu.edu/stat501/lesson/11/11.2>

# %%
display_cooks_distance_plot(results);

# %%
display_DFFITS_plot(results);

# %%
display_hat_leverage_studentized(results)

# %% [markdown]
# - Looking at all three studentized plots for leverage, it can be concluded that even if there are a few outliers, none wield a significant influence on the regression since the points with high leverage values have low studentized residual values.

# %%
allDone()
