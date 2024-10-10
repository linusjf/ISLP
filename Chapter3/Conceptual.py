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
# # Conceptual

# %% [markdown]
# ## Import notebook funcs

# %%
from notebookfuncs import *

# %% [markdown]
# ## 1. Describe the null hypotheses to which the p-values given in Table 3.4 correspond. Explain what conclusions you can draw based on these p-values. Your explanation should be phrased in terms of sales, TV, radio, and newspaper, rather than in terms of the coefficients of the linear model.

# %% [markdown]
# | | Coefficient | Standard Error | t-statistic | p-value |
# |:--| :-----: | :---: | :---: | :--: |
# |Intercept | 2.939 | 0.3119 | 9.12 | < 0.0001 |
# |TV | 0.046 | 0.0014 | 32.81 | < 0.0001 |
# | Radio | 0.189 | 0.0086 | 21.89 | < 0.0001 |
# | Newspaper | -0.001 | 0.0059 | -0.18 | 0.8599 |

# %% [markdown]
# - The null hypotheses to which the given p-values correspond are:
# - $H_0 : \beta_{Intercept} = \beta_{TV} = \beta_{Radio} = \beta_{Newspaper} = 0$
# - The p-values for Intercept, TV and Radio are significant i.e., less that 0.01.
# - Hence, we can reject the null hypotheses that $\beta_{Intercept}$, $\beta_{TV}$ and $\beta_{Radio}$ are zero i.e., the intercept and coefficient values for TV and Radio are significant in the multilinear regression model.
# - The model thus becomes $Sales = 2.939 + 0.0046 * TV + 0.189 * Radio$
# - The Intercept value implies that in the absence of any advertsing spend on TV and Radio, the sales would on average be $2.939 * 1000 = 2939$ units.
# - The coefficient of 0.046 on TV suggests that for every $1000 spent on TV advertising, the sales units increase by 0.0046 * 1000 = 46 units. Radio spend remaining constant.
# - Similarly, for every additional 1000 dollars spent on radio, the sales units increase by 0.189 * 1000 = 189 units keeping TV spending constant.

# %% [markdown]
# ## 2. Carefully explain the differences between the KNN classifier and KNN regression methods.

# %%

# %% [markdown]
# ## 3. Suppose we have a data set with five predictors, $X_1 = GPA, X_2 = IQ, X_3 = Level $ (1 for College and 0 for High School), $X_4 = Interaction$ between GPA and IQ, and $X_5 = Interaction$ between GPA and Level. The response is starting salary after graduation (in thousands of dollars). Suppose we use least squares to fit the model, and get $\beta_0 = 50, \beta_1 = 20, \beta_2 = 0.07, \beta_3 = 35, \beta_4 = 0.01, \beta_5 = −10$.

# %% [markdown]
# ### (a) Which answer is correct, and why?

# %% [markdown]
# #### i. For a fixed value of IQ and GPA, high school graduates earn more, on average, than college graduates.

# %% [markdown]
# #### ii. For a fixed value of IQ and GPA, college graduates earn more, on average, than high school graduates.

# %% [markdown]
# #### iii. For a fixed value of IQ and GPA, high school graduates earn more, on average, than college graduates provided that the GPA is high enough.

# %% [markdown]
# #### iv. For a fixed value of IQ and GPA, college graduates earn more, on average, than high school graduates provided that the GPA is high enough.

# %% [markdown]
# ### (b) Predict the salary of a college graduate with IQ of 110 and a GPA of 4.0.

# %% [markdown]
# ### (c) True or false: Since the coefficient for the GPA/IQ interaction term is very small, there is very little evidence of an interaction effect. Justify your answer.

# %% [markdown]
# ## 4. I collect a set of data (n = 100 observations) containing a single predictor and a quantitative response. I then fit a linear regression model to the data, as well as a separate cubic regression, i.e. $Y = \beta_0 + \beta_1 X + \beta_2 X^2 + \beta_3 X^3 + \epsilon$.

# %% [markdown]
# ### (a) Suppose that the true relationship between X and Y is linear, i.e. $Y = \beta_0 + \beta_1 X + \epsilon$. Consider the training residual sum of squares (RSS) for the linear regression, and also the training RSS for the cubic regression. Would we expect one to be lower than the other, would we expect them to be the same, or is there not enough information to tell? Justify your answer.

# %% [markdown]
# ### (b) Answer (a) using test rather than training RSS.

# %% [markdown]
# ### (c) Suppose that the true relationship between X and Y is not linear, but we don’t know how far it is from linear. Consider the training RSS fo cubic regression. Would we expect one to be lower than the linear regression, and also the training RSS for the cubic regression. Would we expect one to be lower than the other, would we expect them to be the same, or is there not enough information to tell? Justify your answer.

# %% [markdown]
# ### (d) Answer (c) using test rather than training RSS.

# %% [markdown]
# ## 5. Consider the fitted values that result from performing linear regression without an intercept. In this setting, the $i_{th}$ fitted value takes the form 
#
# $ \huge \hat{y_i} = x_i * \hat{\beta} $ 
#
# where
#
# $\huge \hat{\beta} = \huge  \huge (\sum_{i=1}^n x_i y_i) / \huge (\sum_{i^{'}=1}^{n} x_{i^{'}}^2 )$
#
# ### Show that we can write
#
# $\huge \hat{y_i} = \sum_{i^{'}=1}^n a_{i^{'}} y_{i^{'}}$
#
# ### What is $a_{i^{'}}$?
#
# ### Note: We interpret this result by saying that the fitted values from linear regression are linear combinations of the response values.

# %% [markdown]
# ## 6. Using (3.4), argue that in the case of simple linear regression, the least squares line always passes through the point $(\bar{x}, \bar{y})$.

# %% [markdown]
# ## 7. It is claimed in the text that in the case of simple linear regression of Y onto X, the $R^2$ statistic (3.17) is equal to the square of the correlation between X and Y (3.18). Prove that this is the case. For simplicity, you may assume that $\bar{x} = \bar{y} = 0$.

# %%
allDone();
