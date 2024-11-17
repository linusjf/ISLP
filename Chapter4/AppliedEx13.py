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
import seaborn as sns
import matplotlib.pyplot as plt

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

# %% [markdown]
# We can see that the correlation between Year and LogVolume is 0.98 which is much higher than the correlation between Year and Volume which is 0.84. That's because log transformation is non-linear and the original relation was non-linear as seen from the plot below.

# %%
plt.subplot(1,2,1)
plt.plot(np.exp(Weekly["LogVolume"]), label="Volume", c="r");
plt.legend();
plt.subplot(1,2,2)
plt.plot(Weekly["LogVolume"], label="LogVolume");
plt.legend();

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
# We can verify the above conclusion from kurtosis definition by plotting the boxplots for the continuous  variables, Lag1 - Lag5, Today and LogVolume.

# %%
Weekly.boxplot(column=["Lag1","Lag2","Lag3", "Lag4", "Lag5", "Today", "LogVolume"]);

# %%
Weekly["Direction"].value_counts().plot(kind="pie",autopct="%.2f",title="Direction");

# %% [markdown]
# - Thus, we see from the pie-chart, that if we classify all responses as 'Up', we would still achieve an accuracy level of 55.56%. This is the base level which we have to improve upon.

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
