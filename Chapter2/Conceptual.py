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
# ### (d) The variance of the error terms, i.e. σ 2 = Var(ϵ), is extremely high.

# %% [markdown]
# ## 2. Explain whether each scenario is a classification or regression problem, and indicate whether we are most interested in inference or prediction. Finally, provide n and p.

# %% [markdown]
# ### (a) We collect a set of data on the top 500 firms in the US. For each firm we record profit, number of employees, industry and the CEO salary. We are interested in understanding which factors affect CEO salary.

# %% [markdown]
# ### (b) We are considering launching a new product and wish to know whether it will be a success or a failure. We collect data on 20 similar products that were previously launched. For each product we have recorded whether it was a success or failure, price charged for the product, marketing budget, competition price, and ten other variables.

# %% [markdown]
# ### (c) We are interested in predicting the % change in the USD/Euro exchange rate in relation to the weekly changes in the world stock markets. Hence we collect weekly data for all of 2012. For each week we record the % change in the USD/Euro, the %change in the US market, the % change in the British market, and the % change in the German market.

# %%
allDone();
