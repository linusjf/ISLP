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
# ## Exercise 13

# %% [markdown]
# This question should be answered using the Weekly data set, which is part of the ISLP package. This data is similar in nature to the Smarket data from this chapter's lab, except that it contains 1,089 weekly returns for 21 years, from the beginning of 1990 to the end of 2010.

# %% [markdown]
# ### (a)

# %% [markdown]
# Produce some numerical and graphical summaries of the Weekly data. Do there appear to be any patterns?

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
