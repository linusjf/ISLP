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
# - Flexible models, in general, perform better with more data and a smaller number of parameters to data point count.
# - A flexible model fits the data better in the training data.
# - Whether the flexible model performs better on test data depends on the true relationship underlying the data.
# - If the true relationship is linear, then the flexible model will not perform as well on test data as the variance will tend to increase given that these are unseen observations not used to train the model.
# - Generally, the flexible model will be better.

# %% [markdown]
# ### (b) The number of predictors p is extremely large, and the number of observations n is small.

# %% [markdown]
# - The inflexible model will perform better when the number of predictors are extremely large and the number of observations are few in number.
# - The inflexible model will tend to overfit the data.
# - This can be mitigated by use of well-designed syntehtic data but that's a topic yet to be covered or learnt.

# %% [markdown]
# ### (c) The relationship between the predictors and response is highly non-linear.

# %% [markdown]
# - A flexible model will be better since flexible models are better at capturing non-linearities in the data.

# %% [markdown]
# ### (d) The variance of the error terms, i.e.. $\sigma^2 = Var(\epsilon)$, is extremely high.

# %% [markdown]
# - When the variance of the error terms is very high, the flexible model is worse since it will tend to overfit the model i.e.. it will tend to explain part of the variance as well i.e, it will tend to fit too much of the noise.

# %% [markdown]
# ## 2. Explain whether each scenario is a classification or regression problem, and indicate whether we are most interested in inference or prediction. Finally, provide n and p.

# %% [markdown]
# ### (a) We collect a set of data on the top 500 firms in the US. For each firm we record profit, number of employees, industry and the CEO salary. We are interested in understanding which factors affect CEO salary.

# %% [markdown]
# - The CEO Salary is a continuous variable.
# - n = 500
# - p = 3
# - The problem is thus a regression problem.

# %% [markdown]
# ### (b) We are considering launching a new product and wish to know whether it will be a success or a failure. We collect data on 20 similar products that were previously launched. For each product we have recorded whether it was a success or failure, price charged for the product, marketing budget, competition price, and ten other variables.

# %% [markdown]
# - Response variable is Success/ Failure of the new product
# - n = 20
# - p = 13
# - The problem is a classification one.

# %% [markdown]
# ### (c) We are interested in predicting the % change in the USD/Euro exchange rate in relation to the weekly changes in the world stock markets. Hence we collect weekly data for all of 2012. For each week we record the % change in the USD/Euro, the %change in the US market, the % change in the British market, and the % change in the German market.

# %% [markdown]
# - Response is %change in USD/Euro exchange rate. This is a continuous variable.
# - n = 52
# - p = 3
# - The problem is thus a regression problem.

# %% [markdown]
# ## 3. We now revisit the bias-variance decomposition.

# %% [markdown]
# ### (a) Provide a sketch of typical (squared) bias, variance, training error, test error, and Bayes (or irreducible) error curves, on a single plot, as we go from less flexible statistical learning methods towards more flexible approaches. The x-axis should represent the amount of flexibility in the method, and the y-axis should represent the values for each curve. There should be five curves. Make sure to label each one.

# %% [markdown]
# ![Bias Variance Decomposition](Conceptual.jpg)

# %%
def draw_bias_variance_plot():

  import numpy as np
  from matplotlib import pyplot as plt
  from scipy import interpolate

  # Set the figure size
  plt.rcParams["figure.figsize"] = [15.00, 10]
  plt.rcParams["figure.autolayout"] = True

  # Drae Test MSE curve
  nodes = np.array( [ [0, 135], [45, 90], [90, 135] ] )

  x = nodes[:,0]
  y = nodes[:,1]

  tck,u     = interpolate.splprep( [x,y] ,s = 0 , k=2)
  xnew,ynew = interpolate.splev( np.linspace( 0, 1, 100 ), tck,der = 0)

  plt.plot( xnew ,ynew, 'Orange', label="Test MSE" )

  ax = plt.gca()

  # Hide X and Y axes label marks
  ax.xaxis.set_tick_params(labelbottom=False)
  ax.yaxis.set_tick_params(labelleft=False)

  # Hide X and Y axes tick marks
  ax.set_xticks([])
  ax.set_yticks([])

  # set x and y limits
  ax.set_xlim(0, 100)
  ax.set_ylim(0,150)

  plt.xlabel("Model Flexibility")
  plt.ylabel("MSE")

  # Draw irreducible error
  ax.axhline(75, linestyle='--', label="Irreducible error")

  plt.legend()
  plt.show();

draw_bias_variance_plot();

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
# - Flexible approaches to regression or classification are able to capture non-linearities in the data better than a non-flexible (or linear approach).
# - They are able to generalise better to unseen data when the true models are not linear in nature.
# - One of their disadvantages is that they're hungry for data and have a lot more parameters than a simpler, linear approach.
# - Linear approaches work better when the modelers have an idea of the domain and are thus able to generate parsimonious models which are usually good enough for their purposes.
# - Interpretability is also simpler with less flexible models.
# - It is difficult to understand a more flexible model.
#
# A more flexible approach is preferred when the model is known to be complex and cannot be easily captured by a less flexible model.
#
# A less flexible approach is preferred when the modelers seek interpretability, simplicity, speed, eficiency and have domain knowledge and expertise. Addtionally, when data quality is poor, a simpler model may function better than a more flexible model that may not learn as well and which adapts to the noise in the data.
#
# References: 1> <https://www.baeldung.com/cs/ml-flexible-and-inflexible-models>

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
