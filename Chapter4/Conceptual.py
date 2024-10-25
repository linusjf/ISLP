# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.4
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Conceptual: Chapter 4 &mdash; Classification

# %% [markdown]
# ## Import notebook functions

# %%
from notebookfuncs import *

# %% [markdown]
# ## Exercise 1
#
# Using a little bit of algebra, prove that 
# $$
# \Large p(X)  = \frac {e^{\beta_0 + \beta_1 * X}} {1 + {e^{\beta_0 + \beta_1 * X}}}
# $$
#  is equivalent to
# $$
# \Large \frac {p(X)} {1 - p(X)} = e^{\beta_0 + \beta_1 * X}
# $$.
#
# In other words, the logistic function representation and logit representation for the logistic regression model are equivalent. 

# %% [markdown]
# ## Exercise 2
# It was stated in the text that classifying an observation to the class for which
# $$
# \Large p_{k}(x) = \frac {\pi_k \frac {1} {\sqrt{2\pi}\sigma} exp(- \frac {1} {2\sigma^2}(x - \mu_k)^2)} {\sum_{l=1}^K \pi_l \frac {1} {\sqrt{2\pi}  \sigma} exp (- \frac {1} {2\sigma^2}(x - \mu_l)^2)}
# $$
# (4.17)
# is largest is equivalent to classifying an observation to the class for which
# $$
# \Large \delta_k(x) = x.\frac {\mu_k} {\sigma^2} - \frac {\mu_k^2} {2\sigma^2} + log(\pi_k)
# $$
# (4.18) is the largest. Prove that this is the case. In other words, under the assumption that the observations in the $k_{th}$ class are drawn from a $N(\mu_k , \sigma^2)$ distribution, the Bayes classifier assigns an observation to the class for which the discriminant function is maximized.

# %% [markdown]
# ## Exercise 3 
# This problem relates to the QDA model, in which the observations within each class are drawn from a normal distribution with a class specific mean vector and a class specific covariance matrix. We consider the simple case where p = 1; i.e. there is only one feature. Suppose that we have K classes, and that if an observation belongs to the kth class then X comes from a one-dimensional normal distribution, $X ∼ N(\mu_k , \sigma_k^2)$. Recall that the density function for the one-dimensional normal distribution is given in 
# $$
# \Large f_k(x) = \frac {1} {\sqrt{2\pi}\sigma_k} exp \Big (- \frac {1} {2\sigma_k^2}(x - \mu_k)^2 \Big )
# $$
#
# (4.16). Prove that in this case, the Bayes classifier is not linear. Argue that it is in fact quadratic. 
#
# **Hint: For this problem, you should follow the arguments laid out in Section 4.4.1, but without making the assumption that $\sigma_1^2 = · · · = \sigma_K^2$**

# %% [markdown]
# ## Exercise 4
# When the number of features p is large, there tends to be a deterioration in the performance of KNN and other local approaches that perform prediction using only observations that are near the test observation for which a prediction must be made. This phenomenon is known as the curse of dimensionality, and it ties into the fact that non-parametric approaches often perform poorly when p is large. We will now investigate this curse.

# %% [markdown]
# ### (a)
# Suppose that we have a set of observations, each with measurements on p = 1 feature, X. We assume that X is uniformly (evenly) distributed on [0, 1]. Associated with each observation is a response value. Suppose that we wish to predict a test observation’s response using only observations that are within 10 % of the range of X closest to that test observation. For instance, in order to predict the response for a test observation with X = 0.6, we will use observations in the range [0.55, 0.65]. On average, what fraction of the available observations will we use to make the prediction?

# %% [markdown]
# ### (b)
# Now suppose that we have a set of observations, each with measurements on p = 2 features, $X_1$ and $X_2$ . We assume that $(X_1 , X_2 )$ are uniformly distributed on [0, 1] × [0, 1]. We wish to predict a test observation’s response using only observations that are within 10 % of the range of $X_1$ and within 10 % of the range of $X_2$ closest to that test observation. For instance, in order to predict the response for a test observation with $X_1$ = 0.6 and $X_2$ = 0.35, we will use observations in the range [0.55, 0.65] for $X_1$ and in the range [0.3, 0.4] for $X_2$ . On average, what fraction of the available observations will we use to make the prediction?

# %% [markdown]
# ### (c)
# Now suppose that we have a set of observations on p = 100 features. Again the observations are uniformly distributed on each feature, and again each feature ranges in value from 0 to 1. We wish to predict a test observation’s response using observations within the 10% of each feature’s range that is closest to that test observation. What fraction of the available observations will we use to make the prediction?

# %% [markdown]
# ### (d)
# Using your answers to parts (a)–(c), argue that a drawback of KNN when p is large is that there are very few training observations “near” any given test observation.

# %% [markdown]
# ### (e)
# Now suppose that we wish to make a prediction for a test observation by creating a p-dimensional hypercusegment, when p = 2 it is a square, and when p = 100 it is abe centered around the test observation that contains, on average, 10 % of the training observations. For p = 1, 2, and 100, what is the length of each side of the hypercube? Comment on your answer.
#
# **Note: A hypercube is a generalization of a cube to an arbitrary number of dimensions. When p = 1, a hypercube is simply a line segment, when p = 2 it is a square, and when p = 100 it is a 100-dimensional cube.**

# %% [markdown]
# ## Exercise 5
#
# We now examine the differences between LDA and QDA.

# %% [markdown]
# ### (a)
# If the Bayes decision boundary is linear, do we expect LDA or QDA to perform better on the training set? On the test set?

# %% [markdown]
# If the Bayes decision boundary is linear, QDA may perform better on the training set since it may tend to overfit the data.
# However, on the test  set, it is LDA that will fit better to outside data not seen or encountered in the training set.

# %% [markdown]
# ### (b) 
# If the Bayes decision boundary is non-linear, do we expect LDA or QDA to perform better on the training set? On the test set?

# %% [markdown]
# If the Bayes decision boundary is non-linear, QDA will perform better on the training set since it will tend to overfit the data
# However, on the test  set, QDA will definitely fit better to outside data not seen or encountered in the training set.

# %% [markdown]
# ### (c)
# In general, as the sample size n increases, do we expect the test prediction accuracy of QDA relative to LDA to improve, decline, or be unchanged? Why?

# %% [markdown]
# ### (d)
# True or False: Even if the Bayes decision boundary for a given problem is linear, we will probably achieve a superior test error rate using QDA rather than LDA because QDA is flexible enough to model a linear decision boundary. Justify your answer.

# %% [markdown]
# ## Exercise 6
#
# Suppose we collect data for a group of students in a statistics class with variables $\large X_1 = hours \: studied, X_2 = undergrad \: GPA$, and $Y = receive \: an \: A$. We fit a logistic regression and produce estimated coefficient, $\large \beta_0 = −6, \beta_1 = 0.05, \beta_2 = 1$.

# %% [markdown]
# ### (a)
# Estimate the probability that a student who studies for 40 hours and has an undergrad GPA of 3.5 gets an A in the class.

# %% [markdown]
# ### (b)
# How many hours would the student in part (a) need to study to have a 50 % chance of getting an A in the class?

# %% [markdown]
# ## Exercise 7 
#
# Suppose that we wish to predict whether a given stock will issue a dividend this year (“Yes” or “No”) based on X, last year’s percent profit. We examine a large number of companies and discover that the mean value of X for companies that issued a dividend was $\large \overline{X}$ = 10, while the mean for those that didn’t was $\large \overline{X}$ = 0. In addition, the variance of X for these two sets of companies was $\large \sigma^2$ = 36. Finally, 80% of companies issued dividends. Assuming that X follows a normal distribution, predict the probability that a company will issue a dividend this year given that its percentage profit was X = 4 last year.
#
# **Hint: Recall that the density function for a normal random variable is
# $$
# \Large f(x) = \frac {1} {\sqrt {2\pi\sigma^2}} e^{- {(x−µ)^2} / {2\sigma^2}}
# $$. You will need to use Bayes’ theorem.**

# %% [markdown]
# ## Exercise 8
#
# Suppose that we take a data set, divide it into equally-sized training and test sets, and then try out two different classification procedures. First we use logistic regression and get an error rate of 20% on the training data and 30% on the test data. Next we use 1-nearest neighbors (i.e. K = 1) and get an average error rate (averaged over both test and training data sets) of 18%. Based on these results, which method should we prefer to use for classification of new observations? Why?

# %% [markdown]
# ## Exercise 9

# %% [markdown]
# This problem has to do with odds. 

# %% [markdown]
# ### (a)
# On average, what fraction of people with an odds of 0.37 of defaulting on their credit card payment will in fact default? 

# %% [markdown]
# ### (b)
# Suppose that an individual has a 16% chance of defaulting on her credit card payment. What are the odds that she will default?

# %% [markdown]
# ## Exercise 10
# Equation 4.32 derived an expression for $\Large log \Big (\frac {Pr(Y=k | X = x)} {Pr(Y=K | X = x)} \Big )$ in the setting where p > 1, so that the mean for the $k_{th}$ class, $\mu_k$ , is a p-dimensional vector, and the shared covariance $\sum$ is a p × p matrix. However, in the setting with p = 1, (4.32) takes a simpler form, since the means $\large \mu_1 , . . . , \mu_K$ and the variance $\large \sigma^2$ are scalars. In this simpler setting, repeat the calculation in (4.32), and provide expressions for $\large a_k$ and $\large b_{kj}$ in terms of $\large \pi_k$ , $\large \pi_K$ , $\large \mu_k$ , $\large \mu_K$ , and $\large \sigma^2$ .
#
# Equation 4.32:
#
# $$
# \Large log \Big (\frac {Pr(Y=k | X = x)} {Pr(Y=K | X = x)} \Big ) = a_k + \sum_{j=1}^p b_{kj}x_j
# $$

# %% [markdown]
# ## Exercise 11
# Work out the detailed forms of $\large a_k$ , $\large b_{kj}$ , and $\large c_{kjl}$ in 
# $$
# \Large log \Big (\frac {Pr(Y=k | X = x)} {Pr(Y=K | X = x)} \Big ) = a_k + \sum_{j=1}^p b_{kj}x_j + \sum_{j=1}^p\sum_{l=1}^pc_{kjl}x_jx_l
# $$
#
# (4.33). Your answer should involve $\large \pi_k$ , $\large \pi_K$ , $\large \mu_k$ , $\large \mu_K$ , $\large \sum_k$ , and $\large \sum_K$ .

# %% [markdown]
# ## Exercise 12
# Suppose that you wish to classify an observation X ∈ R into apples and oranges. You fit a logistic regression model and find that 
# $$
# \large \hat{Pr}(Y = orange | X = x) = \frac {exp(\hat{\beta_0} + \hat{\beta_1}x)} {1 + exp(\hat{\beta_0} + \hat{\beta_1}x)}
# $$.
#
# Your friend fits a logistic regression model to the same data using the softmax formulation in
# $$
# \large \hat{Pr}(Y = orange | X = x) = \frac {e^{\beta_0 + \beta_1x_1 + .... + \beta_px_p}} {\sum_{l=1}^K{e^{\beta_0 + \beta_1x_1 + .... + \beta_px_p}}}
# $$
# (4.13), and finds that 
# $$
# \large \hat{Pr}(Y = orange | X = x) = \frac {exp({\hat{\alpha}}_{orange0} + {\hat{\alpha}}_{orange1}x)} {exp({\hat{\alpha}}_{orange0} + {\hat{\alpha}}_{orange1}x) + exp({\hat{\alpha}}_{apple0} + {\hat{\alpha}}_{apple1}x)  }
# $$
#

# %% [markdown]
# ### (a)
# What is the log odds of orange versus apple in your model?

# %% [markdown]
# ### (b)
# What is the log odds of orange versus apple in your friend’s model?

# %% [markdown]
# ### (c)
# Suppose that in your model, $\beta_0$ = 2 and $\beta_1$ = −1. What are the coefficient estimates in your friend’s model? Be as specific as possible.

# %% [markdown]
# ### (d)
# Now suppose that you and your friend fit the same two models on a different data set. This time, your friend gets the coefficient estimates $\large {\hat{\alpha}}_{orange0}$ = 1.2, $\large {\hat{\alpha}}_{orange1}$ = −2, $\large {\hat{\alpha}}_{orange0}$ = 3, $\large {\hat{\alpha}}_{orange1}$ = 0.6. What are the coefficient estimates in your model?

# %% [markdown]
# ### (e) 
# Finally, suppose you apply both models from (d) to a data set with 2,000 test observations. What fraction of the time do you expect the predicted class labels from your model to agree with those from your friend’s model? Explain your answer.

# %%
allDone();
