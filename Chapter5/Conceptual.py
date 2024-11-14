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
# # Conceptual Exercises

# %% [markdown]
# # Import notebook functions

# %%
from notebookfuncs import *

# %% [markdown]
# ## Exercise 1
#
# **Using basic statistical properties of the variance, as well as single-variable calculus, derive
# $$
# \large \alpha = \frac {\sigma_Y^2 - \sigma_{XY}^2 } {\sigma_X^2 + \sigma_Y^2 - 2\sigma_{XY}^2 }
# $$.**
#
# **In other words, prove that $\alpha$ given by the equation above does indeed minimize $Var(\alpha X + (1 − \alpha)Y)$.**

# %% [markdown]
# By statistical properties of the variance, we have
# $$\begin{aligned}
# & Var(aX + bY) = Var(aX) + Var(bY) + 2Cov(aX, bY) \\
# & = a^2Var(X) + b^2Var(Y) + 2abCov(X,Y) \\
# & \therefore Var(\alpha X + (1 − \alpha)Y) \\
# & = \alpha^2Var(X) + (1 - \alpha)^2Var(Y) + 2\alpha(1 - \alpha)Cov(X,Y) \\
# & = \alpha^2Var(X) + Var(Y) + \alpha^2Var(Y) -2 \alpha Var(Y)  + 2 \alpha Cov(X.Y) - 2 \alpha^2Cov(X,Y) \\ 
# & = \alpha^2Var(X) + \alpha^2Var(Y) - 2 \alpha^2Cov(X,Y) -2 \alpha Var(Y)  + 2 \alpha Cov(X.Y) + Var(Y) \\
# & = \alpha^2(Var(X) + Var(Y) - 2 Cov(X,Y)) -2 \alpha (Var(Y) - Cov(X,Y)) + Var(Y)
# \end{aligned}$$

# %% [markdown]
# Using single-variable calculus and differentiating the above equation w.r.t $\alpha$ and equating it to zero to discover the value of $\alpha$ that minimizes the variance, we have
#
# $$\begin{aligned}
# & 2 \alpha (Var(X) + Var(Y) - 2 Cov(X,Y)) -2 (Var(Y) - Cov(X,Y)) = 0 \\
# & \implies 2 \alpha (Var(X) + Var(Y) - 2 Cov(X,Y)) = 2 (Var(Y) - Cov(X,Y)) \\
# & \implies \alpha (Var(X) + Var(Y) - 2 Cov(X,Y)) = Var(Y) - Cov(X,Y) \\
# & \implies \alpha = \frac {Var(Y) - Cov(X,Y)} {Var(X) + Var(Y) - 2 Cov(X,Y)} \\
# & \implies \alpha = \frac {\sigma_Y^2 - \sigma_{XY}^2} {\sigma_X^2 + \sigma_Y^2 - 2\sigma_{XY}^2}
# \end{aligned}$$

# %% [markdown]
# ## Exercise 2
#
# **We will now derive the probability that a given observation is part of a bootstrap sample. Suppose that we obtain a bootstrap sample from a set of n observations.**

# %% [markdown]
# ### (a)
# **What is the probability that the first bootstrap observation is not the $j_{th}$ observation from the original sample? Justify your answer.**

# %% [markdown]
# We are obtaining an observation from a set of n observations. $\therefore$ the probability that the first bootstrap observation is the $j_{th}$ observation is $\frac {1} {n}$ . $\therefore$ the probability that it is not is $1 - \frac {1} {n}$.

# %% [markdown]
# ### (b) 
# **What is the probability that the second bootstrap observation is not the $j_{th}$ observation from the original sample?**

# %% [markdown]
# We are sampling with replacement since bootstrap implies the same, the probability that the second observation is not the $j_{th}$ observation is $(1 - \frac{1} {n})^2$.

# %% [markdown]
# ### (c) 
# **Argue that the probability that the $j_{th}$ observation is not in the bootstrap sample is $(1 − \frac {1} {n})^n$.**

# %% [markdown]
# Thus, the probability that the $j_{th}$ observation is not in the entire bootstrap sample of size n is $(1 − \frac {1} {n})^n$.

# %% [markdown]
# ### (d) 
# **When n = 5, what is the probability that the $j_{th}$ observation is in the bootstrap sample?**

# %%
import sympy
from sympy import Symbol, N
n = Symbol('n')
prob_j = 1 - (1 - 1 / n) ** n
N(prob_j.subs(n, 5))

# %% [markdown]
# ### (e) 
# **When n = 100, what is the probability that the $j_{th}$ observation is in the bootstrap sample?**

# %%
N(prob_j.subs(n, 100))

# %% [markdown]
# ### (f)
# **When n = 10, 000, what is the probability that the $j_{th}$ observation is in the bootstrap sample?**

# %%
N(prob_j.subs(n, 10000))

# %% [markdown]
# ### (g) 
# **Create a plot that displays, for each integer value of n from 1 to 100, 000, the probability that the $j_{th}$ observation is in the bootstrap sample. Comment on what you observe.**

# %%
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import mode
probs = np.zeros(100000)
for i in range(100000):
  n = i + 1
  probs[i] = 1 - (1 - 1 / n) ** n
plt.title("Probability that the $j_{th}$ observation is in the bootstrap sanple.")
plt.xlabel("Sample size")
plt.ylabel("Probability");
plt.plot(probs);

# %%
printmd("Here, we can see that as the size of the bootstrap sample increases, the probability of selecting the $j_{th}$ samples stabilizes at ")
printmd(f"**{mode(probs)[0]:.4f}**.")

# %% [markdown]
# ### (h)
# **We will now investigate numerically the probability that a bootstrap sample of size n = 100 contains the $j_{th}$ observation. Here j = 4. We first create an array store with values that will subsequently be overwritten using the function `np.empty()`. We then repeatedly create bootstrap samples, and each time we record whether or not the fifth observation is contained in the bootstrap sample.**
#
# ```python
# rng = np.random.default_rng(10)
# store = np.empty(10000)
# for i in range(10000):
#   store[i] = np.sum(rng.choice(100, size=100, replace=True) == 4) > 0
# np.mean(store)
# ```
# **Comment on the results obtained.**

# %%
rng = np.random.default_rng(10)
store = np.empty(10000)
for i in range(10000):
  # This stores the number of bootstrap samples that contain 4
  store[i] = np.sum(rng.choice(100, size=100, replace=True) == 4) > 0
np.mean(store)

# %% [markdown]
# *Note: We had to change the code above so that 100 numbers are sampled from the range 0 - 99 and thus 10000 bootstrap samples are generated. The earlier code only sampled 1 element at a time.*

# %%
printmd(f"Thus, we see that the probability closely matches what we obtained theoretically at {np.mean(store)}.")

# %% [markdown]
# ## Exercise 3
# **We now review k-fold cross-validation.**

# %% [markdown]
# ### (a) 
# **Explain how k-fold cross-validation is implemented.**

# %% [markdown]
# ### (b) 
# **What are the advantages and disadvantages of k-fold cross-validation relative to:**
#
#  **i.  The validation set approach?**
# **ii.  LOOCV?**

# %% [markdown]
# ## Exercise 4
# **Suppose that we use some statistical learning method to make a prediction for the response Y for a particular value of the predictor X. Carefully describe how we might estimate the standard deviation of our prediction.**

# %%
allDone();
