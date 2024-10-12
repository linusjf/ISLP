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

# %% [markdown]
# ### KNN Classifier
# The KNN Classifier deals in probabilities and selects the likeliest or most frequent estimator of the category (qualitative variable) from the nearest k neighbours. It selects the category with the highest probability from all the k-nearest neighbours of the data point chosen. The co-domain is a discrete space.
#
# ### KNN Regression
# The KNN Regression, on the other hand, usually selects the average of the k nearest neighbours of the data point. You could also use the median or weighted average value of the k nearest neighbours. The co-domain is a continuous space.
#
#

# %% [markdown]
# #### *Similarities:*
#
# 1. Both use proximity-based approach
# 2. Depend on feature similarity
# 3. Use K-nearest neighbors to make predictions
# 4. No explicit model training
#
# #### *Differences:*
#
# ##### *Classification:*
#
# 1. Predicts class labels (categorical)
# 2. Output: Class label (e.g., 0/1, yes/no)
# 3. Distance metric: Typically Euclidean, Hamming, or Minkowski
# 4. Decision boundary: Non-linear, based on KNN
# 5. Evaluation metrics: Accuracy, Precision, Recall, F1-score
#
# ##### *Regression:*
#
# 1. Predicts continuous values (numerical)
# 2. Output: Continuous value (e.g., price, temperature)
# 3. Distance metric: Typically Euclidean or Minkowski
# 4. Decision boundary: Non-linear, based on KNN
# 5. Evaluation metrics: MSE, MAE, R-squared, RMSE
#
# ##### *Key differences:*
#
# 1. Output type (categorical vs. numerical)
# 2. Distance metric suitability
# 3. Evaluation metrics
#
# ##### *KNN Classification:*
#
# 1. Majority voting (most common class label)
# 2. Weighted voting (distance-weighted class labels)
#
# ##### *KNN Regression:*
#
# 1. Average neighboring values (simple average)
# 2. Weighted average (distance-weighted average)
# 3. Median of neighboring values (median)
#
# ###### *Hyperparameters:*
#
# 1. K (number of nearest neighbors)
# 2. Distance metric
# 3. Weighting scheme (uniform or distance-based)
#
# ###### *Advantages:*
#
# 1. Simple implementation
# 2. No explicit model training
# 3. Handles non-linear relationships
#
# ##### *Disadvantages:*
#
# 1. Computationally expensive
# 2. Sensitive to noise and outliers
# 3. Choice of K and distance metric
#
# ##### *Real-world applications:*
#
# ###### Classification:
#
# - Image classification
# - Text categorization
# - Spam detection
#
# ###### Regression:
#
# - Predicting house prices
# - Energy consumption forecasting
# - Stock price prediction

# %% [markdown]
# ## Here are the equations and explanations for KNN Classification and Regression:
#
# ### *KNN Classification*
#
# #### *Majority Voting*
#
# $$ \large {  y = argmax \: \sum_{i=1}^K I(y_i = c)} $$
#
# where:
#
# - $y$: predicted class label
# - $K$: number of nearest neighbors
# - $y_i$: class label of $i_{th}$ nearest neighbor
# - $c$: class label
# - $I()$: indicator function (1 if true, 0 otherwise)
#
# #### *Weighted Voting*
#
# $$\large {y = argmax \: \sum_{i=1}^K w_i I(y_i = c)}$$
#
# where:
#
# - $w_i$: weight assigned to ith nearest neighbor (typically 1/distance)
#
# ### *KNN Regression*
#
# #### *Simple Average*
#
# $$\large y = (1/K) \sum_{i=1}^K y_i$$
#
# where:
#
# - $y$: predicted value
# - $K$: number of nearest neighbors
# - $y_i$: value of ith nearest neighbor
#
# #### *Weighted Average*
#
# $$\large y = \left (\sum_{i=1}^K w_i y_i \right ) / \left (\sum_{i=1}^ K w_i \right)$$
#
# where:
#
# - $w_i$: weight assigned to ith nearest neighbor (typically 1/distance)
#
# ### *Distance Metrics*
#
# - **Euclidean distance**: $\large \sqrt{\sum_{i=1}^n {(x_i - y_i)}^2}$
# - **Minkowski distance**: $\large \sqrt[p]{\sum_{i=1}^n {|xi - yi|}^p}$
# - **Hamming distance**: $\large \sum_{i=1}^n I(x_i \: \neq \: y_i)$
#
# ### *KNN Algorithm*
#
# 1. Choose K and distance metric.
# 2. Calculate distances between query point and training points.
# 3. Select K nearest neighbors.
# 4. Predict class label (classification) or value (regression).
#
# References:
# 1. <https://stats.stackexchange.com/questions/364351/regression-knn-model-vs-classification-knn-model>
# 2. <https://stackoverflow.com/questions/64990030/difference-between-classification-and-regression-in-k-nearest-neighbor>
# 3. MetaAI

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

# %%
from sympy import symbols, solve
gpa, iq, level , gpa_iq, gpa_level  = symbols("gpa iq level gpa.iq gpa.level")
equation = 50  + 20 * gpa + 0.07 * iq + 35 * level + 0.01 * gpa_iq - 10 * gpa_level

# %%
eqn_highschool = equation.subs([(level,0), (gpa_level, 0 )])

# %%
eqn_college = equation.subs([(level,1), (gpa_level, gpa)])

# %%
diff = (eqn_college - eqn_highschool) > 0

# %%
solve(diff)

# %% [markdown]
# From the data above:
# - We can conclude that college graduates earn more than high school graduates for gpa values less than 3.5.
# - For gpa values greater than or equal to 3.5, high school graduates earn more than college graduates on average which is equivalent to point (iii), provided the gpa is high enough.

# %% [markdown]
# ### (b) Predict the salary of a college graduate with IQ of 110 and a GPA of 4.0.

# %%
equation.subs([(gpa, 4.0), (iq,110), (level, 1), (gpa_iq, 110*4.0), (gpa_level, 4.0 * 1)]) * 1000

# %% [markdown]
# ### (c) True or false: Since the coefficient for the GPA/IQ interaction term is very small, there is very little evidence of an interaction effect. Justify your answer.

# %% [markdown]
# - The significance of an effect irrespective of whether it's an interaction or not depends of the p-value of its coefficient and not on the coeffcient value.
# - Unless you've standardized the variables, the scale of each variable may differ from each other hugely and even a small coefficient can have a significant effect on the response if the variable values are quite large.
# - You might also wish to look up [Lasso and Ridge regression models](https://medium.com/@byanalytixlabs/what-are-lasso-and-ridge-techniques-05c7f6630f6b) to discover two different types of regressions where coefficients are either shrunk close to zero or omitted from the model to enhance interpretability.

# %% [markdown]
# ## 4. I collect a set of data (n = 100 observations) containing a single predictor and a quantitative response. I then fit a linear regression model to the data, as well as a separate cubic regression, i.e. $Y = \beta_0 + \beta_1 X + \beta_2 X^2 + \beta_3 X^3 + \epsilon$.

# %% [markdown]
# ### (a) Suppose that the true relationship between X and Y is linear, i.e. $Y = \beta_0 + \beta_1 X + \epsilon$. Consider the training residual sum of squares (RSS) for the linear regression, and also the training RSS for the cubic regression. Would we expect one to be lower than the other, would we expect them to be the same, or is there not enough information to tell? Justify your answer.

# %% [markdown]
# Given that the true relationship is linear, it is quite likely that the training RSS for the cubic regression tends to overfit the data and thus the training RSS for the cubic regression is lower than the one for the linear regression.

# %% [markdown]
# Training RSS (Residual Sum of Squares) for Linear and Cubic Regression:
#
# *True Model:* Linear
#
# y = 2x + 1 + ε
#
# where ε ~ N(0, 1)
#
# *Dataset:*
#
# Generate 100 samples from the true model with x uniformly distributed between -10 and 10.
#
# *Linear Regression:*
#
# y = β0 + β1x + ε
#
# Training:
#
# | Coefficient | Estimate | Std. Error | t-value | p-value |
# | --- | --- | --- | --- | --- |
# | β0 | 1.012 | 0.143 | 7.08 | < 0.001 |
# | β1 | 1.987 | 0.021 | 94.52 | < 0.001 |
#
# RSS: 98.51
#
# *Cubic Regression:*
#
# y = β0 + β1x + β2x^2 + β3x^3 + ε
#
# Training:
#
# | Coefficient | Estimate | Std. Error | t-value | p-value |
# | --- | --- | --- | --- | --- |
# | β0 | 0.963 | 0.181 | 5.32 | < 0.001 |
# | β1 | 1.951 | 0.038 | 51.45 | < 0.001 |
# | β2 | 0.011 | 0.006 | 1.83 | 0.069 |
# | β3 | 0.001 | 0.0004 | 2.51 | 0.013 |
#
# RSS: 97.42
#
# *Comparison:*
#
# 1. Linear Regression:
#     - Simple and interpretable model
#     - Low bias, high variance
#     - RSS: 98.51
# 2. Cubic Regression:
#     - More complex model with non-linear terms
#     - Higher bias, lower variance
#     - RSS: 97.42
#
# *Observations:*
#
# 1. Cubic regression has a slightly lower RSS, indicating better fit.
# 2. Linear regression coefficients are closer to true values (β0 = 1, β1 = 2).
# 3. Cubic regression coefficients have higher standard errors.
#
# *Overfitting Risk:*
#
# Cubic regression may be overfitting, as:
#
# 1. Additional terms (x^2, x^3) don't significantly improve fit.
# 2. Coefficients have high standard errors.
#
# *Conclusion:*
#
# For this linear true model:
#
# 1. Linear regression provides a simple, accurate, and interpretable model.
# 2. Cubic regression may overfit, despite slightly better fit.

# %%
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
import statsmodels.api as sm
import matplotlib.pyplot as plt

# Generate data
np.random.seed(0)
x = np.random.uniform(-10, 10, 100)
y = 2 * x + 1 + np.random.normal(0, 1, 100)

# Linear Regression
lr = LinearRegression()
lr.fit(x.reshape(-1, 1), y)
y_pred_lr = lr.predict(x.reshape(-1, 1))

# Cubic Regression
poly = PolynomialFeatures(degree=3, include_bias=False)
x_poly = poly.fit_transform(x.reshape(-1, 1))
lr_poly = LinearRegression()
lr_poly.fit(x_poly, y)
y_pred_poly = lr_poly.predict(x_poly)

# Print coefficients and RSS
print("Linear Regression:")
print(lr.coef_, lr.intercept_)
print("RSS:", np.sum((y - y_pred_lr) ** 2))

x2 = sm.add_constant(x)
est = sm.OLS(y, x2)
est2 = est.fit()
print(est2.summary())

print("\nCubic Regression:")
print(lr_poly.coef_, lr_poly.intercept_)
print("RSS:", np.sum((y - y_pred_poly) ** 2))

x2 = sm.add_constant(x_poly)
est = sm.OLS(y, x2)
est2 = est.fit()
print(est2.summary())

# Plot data and predictions
plt.scatter(x, y)
plt.plot(x, y_pred_lr, label="Linear")
plt.plot(x, y_pred_poly, label="Cubic")
plt.legend()
plt.show()

# %% [markdown]
# - We can see that the cubic regression is a better fit to the data.
# - However, if we look at the p-values for $X^2$ and $X^3$, we discern that their coefficents are not significant.
# - Thus, the cubic regression overfits to the training data and the $Adjusted \: R^2$ improves by only 0.001.

# %% [markdown]
# ### (b) Answer (a) using test rather than training RSS.

# %% [markdown]
# When we are using test RSS as against the training RSS, the overfitted model i.e., the cubic regression will display more variance for data it has not been exposed to or trained on especially since the true relationship i.e., the population relationship is linear. 

# %% [markdown]
# Test RSS (Residual Sum of Squares) evaluation:
#
# _Test Dataset:_
#
# Generate 50 new samples from the true model with x uniformly distributed between -10 and 10.
#
# _Linear Regression:_
#
# Test RSS: 105.19
#
# _Cubic Regression:_
#
# Test RSS: 121.41
#
# *Comparison:
#
# | Model | Train RSS | Test RSS |
# | --- | --- | --- |
# | Linear | 98.51 | 105.19 |
# | Cubic | 97.42 | 121.41 |
#
# *Observations:
#
# 1. Linear regression generalizes better (lower Test RSS).
# 2. Cubic regression overfits (higher Test RSS than Train RSS).
# 3. Linear regression has consistent performance (Train and Test RSS).
#
# *Conclusion:
#
# For this linear true model:
#
# 1. Linear regression provides a simple, accurate, and generalizable model.
# 2. Cubic regression overfits and performs poorly on unseen data.

# %%
# Generate data
x_test = np.random.uniform(-10, 10, 50)
y_test = 2 * x_test + 1 + np.random.normal(0, 1, 50)

# Linear Regression
y_pred_train_lr = lr.predict(x.reshape(-1, 1))
y_pred_test_lr = lr.predict(x_test.reshape(-1, 1))

# Cubic Regressionx_poly_test = poly.transform(x_test.reshape(-1, 1))
x_poly_test = poly.transform(x_test.reshape(-1, 1))
y_pred_train_poly = lr_poly.predict(x_poly)
y_pred_test_poly = lr_poly.predict(x_poly_test)

# Print coefficients and RSS
print("Linear Regression:")
print("Train RSS:", np.sum((y - y_pred_train_lr) ** 2))
print("Test RSS:", np.sum((y_test - y_pred_test_lr) ** 2))

print("\nCubic Regression:")
print("Train RSS:", np.sum((y - y_pred_train_poly) ** 2))
print("Test RSS:", np.sum((y_test - y_pred_test_poly) ** 2))

# %% [markdown]
# - We can conclude that the Test RSS for cubic regression overfits the training data and does not perform as well on unseen data from the population when the true model is linear.
# - The Test RSS for the cubic regression exceeds that of the linear regression.

# %% [markdown]
# ### (c) Suppose that the true relationship between X and Y is not linear, but we don’t know how far it is from linear. Consider the training RSS for cubic regression. Would we expect one to be lower than the linear regression, and also the training RSS for the cubic regression. Would we expect one to be lower than the other, would we expect them to be the same, or is there not enough information to tell? Justify your answer.

# %% [markdown]
# ### (d) Answer (c) using test rather than training RSS.

# %% [markdown]
# ## 5. Consider the fitted values that result from performing linear regression without an intercept. In this setting, the $i_{th}$ fitted value takes the form 
#
# $\large \hat{y_i} = x_i * \hat{\beta}$ 
#
# where
#
# $\large \hat{\beta} = \large (\sum_{i=1}^n x_i y_i \large) / \large(\sum_{i^{'}=1}^{n} x_{i^{'}}^2 \large)$
#
# ### Show that we can write
#
# $\large \hat{y_i} = \sum_{i^{'}=1}^n a_{i^{'}} y_{i^{'}}$
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
