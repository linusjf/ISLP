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
# # Applied: Exercise 13

# %% [markdown]
# # Import notebook funcs

# %%
from notebookfuncs import *

# %% [markdown]
# ## Import libraries

# %%
from ISLP import load_data
from summarytools import dfSummary
import numpy as np
import klib

# %% [markdown]
# ## Exercise 13

# %% [markdown]
# This question should be answered using the Weekly data set, which is part of the ISLP package. This data is similar in nature to the Smarket data from this chapter's lab, except that it contains 1,089 weekly returns for 21 years, from the beginning of 1990 to the end of 2010.

# %%
Weekly = load_data("Weekly")
Weekly["LogVolume"] = np.log(Weekly["Volume"])
Weekly = Weekly.drop(columns=["Volume"])
Weekly = klib.convert_datatypes(Weekly)
Weekly.head()

# %% [markdown]
# *We transform column Volume to LogVolume since this is the most symmetrical among the transformations sqrt, sqrt4 and log (as evidenced by its low skew value).*

# %%
Weekly.shape

# %% [markdown]
# ### (a)

# %% [markdown]
# Produce some numerical and graphical summaries of the Weekly data. Do there appear to be any patterns?

# %%
dfSummary(Weekly)

# %%
klib.corr_plot(Weekly);

# %%
klib.dist_plot(Weekly);

# %% [markdown]
# #### Skewness
#
# Skewness is a measure of asymmetry or distortion of symmetric distribution. It measures the deviation of the given distribution of a random variable from a symmetric distribution, such as normal distribution. A normal distribution is without any skewness, as it is symmetrical on both sides.

# %% [markdown]
# #### Kurtosis
#
# Negative kurtosis, also known as platykurtic, is a measure of a distribution's thin tails, meaning that outliers are infrequent:
#
# ##### Explanation
# Kurtosis is a statistical measure that describes the shape of a distribution's tails in relation to its overall shape. It measures how often outliers occur, or the "tailedness" of the distribution.
#
# ##### Kurtosis types
# A distribution with a kurtosis of 3 is considered mesokurtic, meaning it has a medium tail. A distribution with a kurtosis greater than 3 is leptokurtic, meaning it has a fat tail and a lot of outliers. A distribution with a kurtosis less than 3 is platykurtic, meaning it has a thin tail and infrequent outliers.
#
# ##### Kurtosis vs peakedness
# Kurtosis measures "tailedness," not "peakedness". A distribution can have a lower peak with high kurtosis, or a sharply peaked distribution with low kurtosis.
#
# ##### Calculating kurtosis
# Kurtosis is calculated mathematically as the standardized fourth moment of a distribution.

# %% [markdown]
# ### (b)

# %% [markdown]
# Use the full data set to perform a logistic regression with Direction as the response and the five lag variables plus Volume as predictors. Use the summary function to print the results. Do any of the predictors appear to be statistically significant? If so, which ones?

# %% [markdown]
# ## (c)

# %% [markdown]
# Compute the confusion matrix and overall fraction of correct predictions. Explain what the confusion matrix is telling you about the types of mistakes made by logistic regression.

# %% [markdown]
# ### (d)

# %% [markdown]
# Now fit the logistic regression model using a training data period from 1990 to 2008, with Lag2 as the only predictor. Compute the confusion matrix and the overall fraction of correct predictions for the held out data (that is, the data from 2009 and 2010).

# %% [markdown]
# ### (e)

# %% [markdown]
# Repeat (d) using LDA.

# %% [markdown]
# ### (f)

# %% [markdown]
# Repeat (d) using QDA.

# %% [markdown]
# ### (g)

# %% [markdown]
# Repeat (d) using KNN with K = 1.

# %% [markdown]
# ### (h)

# %% [markdown]
# Repeat (d) using naive Bayes.

# %% [markdown]
# ### (i)

# %% [markdown]
# Which of these methods appears to provide the best results on this data?

# %% [markdown]
# ### (j)

# %% [markdown]
# Experiment with different combinations of predictors, including possible transformations and interactions, for each of the methods. Report the variables, method, and associated confusion matrix that appears to provide the best results on the held out data. Note that you should also experiment with values for K in the KNN classifier.

# %%
allDone();
