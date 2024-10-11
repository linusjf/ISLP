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
# References:
# 1. <https://stats.stackexchange.com/questions/364351/regression-knn-model-vs-classification-knn-model>
# 2. <https://stackoverflow.com/questions/64990030/difference-between-classification-and-regression-in-k-nearest-neighbor>

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
# Here are the equations and explanations for KNN Classification and Regression:
#
# ### *KNN Classification*
#
# #### *Majority Voting*
#
# $\Huge y = argmax \: \sum_{i=1}^K I(y_i = c)$
#
# where:
#
# - $\Huge y$: predicted class label
# - $\Huge K$: number of nearest neighbors
# - $\Huge y_i$: class label of $i_{th}$ nearest neighbor
# - $\Huge c$: class label
# - $\Huge I()$: indicator function (1 if true, 0 otherwise)
#
# #### *Weighted Voting*
#
# $\Huge y = argmax \: \sum_{i=1}^K w_i I(y_i = c)$
#
# where:
#
# - $\Huge w_i$: weight assigned to ith nearest neighbor (typically 1/distance)
#
# ### *KNN Regression*
#
# #### *Simple Average*
#
# $\Huge y = (1/K) \sum_{i=1}^K y_i$
#
# where:
#
# - $\Huge y$: predicted value
# - $\Huge K$: number of nearest neighbors
# - $\Huge y_i$: value of ith nearest neighbor
#
# #### *Weighted Average*
#
# $\Huge y = (\sum_{i=1}^K w_i y_i) / (\sum_{i=1}^ K w_i)$
#
# where:
#
# - $\Huge w_i$: weight assigned to ith nearest neighbor (typically 1/distance)
#
# ### *Distance Metrics*
#
# - **Euclidean distance**: $\Huge \sqrt{\sum_{i=1}^n {(x_i - y_i)}^2}$
# - **Minkowski distance**: $\Huge (\sum_{i=1}^n {|xi - yi|}^p)^{\normalsize ({1}/{p})}$
# - **Hamming distance**: $\Huge \sum_{i=1}^n I(xi ≠ yi)$
#
# ### *KNN Algorithm*
#
# 1. Choose K and distance metric.
# 2. Calculate distances between query point and training points.
# 3. Select K nearest neighbors.
# 4. Predict class label (classification) or value (regression).
#

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
