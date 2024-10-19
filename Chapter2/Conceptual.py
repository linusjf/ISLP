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

# %%
from IPython.core.interactiveshell import InteractiveShell

InteractiveShell.ast_node_interactivity = "all"

## Import up sound alert dependencies
from IPython.display import Audio, display


def allDone():
    url = "https://sound.peal.io/ps/audios/000/064/733/original/youtube_64733.mp3"
    display(Audio(url=url, autoplay=True))


# %% [markdown]
# ## 1. For each of parts (a) through (d), indicate whether we would generally expect the performance of a flexible statistical learning method to be better or worse than an inflexible method. Justify your answer.

# %% [markdown]
# ### (a) The sample size n is extremely large, and the number of predictors p is small.

# %% [markdown]
# ### (b) The number of predictors p is extremely large, and the number of observations n is small.

# %% [markdown]
# ### (c) The relationship between the predictors and response is highly non-linear.

# %% [markdown]
# ### (d) The variance of the error terms, i.e.. $\sigma^2 = Var(\epsilon)$, is extremely high.

# %% [markdown]
# ## 2. Explain whether each scenario is a classification or regression problem, and indicate whether we are most interested in inference or prediction. Finally, provide n and p.

# %% [markdown]
# ### (a) We collect a set of data on the top 500 firms in the US. For each firm we record profit, number of employees, industry and the CEO salary. We are interested in understanding which factors affect CEO salary.

# %% [markdown]
# ### (b) We are considering launching a new product and wish to know whether it will be a success or a failure. We collect data on 20 similar products that were previously launched. For each product we have recorded whether it was a success or failure, price charged for the product, marketing budget, competition price, and ten other variables.

# %% [markdown]
# ### (c) We are interested in predicting the % change in the USD/Euro exchange rate in relation to the weekly changes in the world stock markets. Hence we collect weekly data for all of 2012. For each week we record the % change in the USD/Euro, the %change in the US market, the % change in the British market, and the % change in the German market.

# %% [markdown]
# ## 3. We now revisit the bias-variance decomposition.

# %% [markdown]
# ### (a) Provide a sketch of typical (squared) bias, variance, training error, test error, and Bayes (or irreducible) error curves, on a single plot, as we go from less flexible statistical learning methods towards more flexible approaches. The x-axis should represent the amount of flexibility in the method, and the y-axis should represent the values for each curve. There should be five curves. Make sure to label each one.

# %% [markdown]
# ### (b) Explain why each of the five curves has the shape displayed in part (a).

# %% [markdown]
# ## 4. You will now think of some real-life applications for statistical learning.

# %% [markdown]
# ### (a) Describe three real-life applications in which classification might be useful. Describe the response, as well as the predictors. Is the goal of each application inference or prediction? Explain your answer.await

# %% [markdown]
# ### (b) Describe three real-life applications in which regression might be useful. Describe the response, as well as the predictors. Is the goal of each application inference or prediction? Explain your answer.

# %% [markdown]
# ### (c) Describe three real-life applications in which cluster analysis might be useful.

# %% [markdown]
# ## 5. What are the advantages and disadvantages of a very flexible (versus a less flexible) approach for regression or classification? Under what circumstances might a more flexible approach be preferred to a less flexible approach? When might a less flexible approach be preferred?

# %% [markdown]
# ## 6. Describe the differences between a parametric and a non-parametric statistical learning approach. What are the advantages of a parametric approach to regression or classification (as opposed to a non-parametric approach)? What are its disadvantages?

# %% [markdown]
# ## 7. The table below provides a training data set containing six observations, three predictors, and one qualitative response variable.
#
# | Obs | $X_1$ | $X_1$ | $X_1$ | $Y$ |
# | --- | --- | --- | --- | --- |
# | 1 | 0 | 3 | 0 | Red |
# | 2 | 2 | 0 | 0 | Red |
# | 3 | 0 | 1 | 3 | Red |
# | 4 | 0 | 1 | 2 |  Green |
# | 5 | -1 | 0 | 1 | Green |
# | 6 | 1 | 1 | 1 | Green |

# %% [markdown]
# ### Suppose we wish to use this data set to make a prediction for Y when $X_1 = X_2 = X-3 = 0$ using K-nearest neighbors.

# %% [markdown]
# #### (a) Compute the Euclidean distance between each observation and the test point, $X_1 = X_2 = X_3 = 0$.

# %% [markdown]
# #### (b) What is our prediction with K = 1? Why?

# %% [markdown]
# #### (c) What is our prediction with K = 3? Why?

# %% [markdown]
# #### (d) If the Bayes decision boundary in this problem is highly non-linear, then would we expect the best value for K to be large or small? Why?

# %%
allDone();
