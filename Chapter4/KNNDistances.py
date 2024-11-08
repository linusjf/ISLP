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
# # KNN Distances

# %% [markdown]
# ## Import notebook funcs

# %%
from notebookfuncs import *

# %%
import torch
import matplotlib.pyplot as plt
import seaborn as sns

def generate_n_dimensional_tensor(n, num_points=1000):
  """Generates an n-dimensional tensor of random numbers.

  Args:
    n: The desired number of dimensions.
    num_points: The number of data points to generate.

  Returns:
    An n-dimensional PyTorch tensor.
  """

  return torch.rand(num_points, n)

def euclidean_distance(p1, p2):
  """Calculates the Euclidean distance between two tensors.

  Args:
    p1: The first tensor.
    p2: The second tensor.

  Returns:
    The Euclidean distance between the two tensors.
  """

  return torch.norm(p1 - p2, dim=-1)

def plot_distance_histograms(distances_list, dimensions):
  """Plots histograms of the Euclidean distances in a grid.

  Args:
    distances_list: A list of lists, each containing distances for a specific dimension.
    dimensions: A list of dimensions.
  """

  fig, axes = plt.subplots(nrows=3, ncols=2, figsize=(12, 12))
  fig.suptitle("Kernel Density Estimates of Euclidean Distances")

  for i, (distances, n) in enumerate(zip(distances_list, dimensions)):
    row = i // 2
    col = i % 2
    ax = axes[row, col]
    sns.kdeplot(distances, fill=True, ax=ax)  # Use fill=True instead of shade=True
    ax.set_xlabel("Euclidean Distance")
    ax.set_ylabel("Density")
    ax.set_title(f"n = {n}")

    min_distance, max_distance = min(distances), max(distances)
    ax.set_xlim(min_distance, max_distance)  # Set x-axis limits for each plot
    ax.text(0.5, 0.9, f"Distance Range: [{min_distance:.4f}, {max_distance:.4f}]",
            transform=ax.transAxes, ha='center', va='top')

  plt.tight_layout()
  plt.show()

# Define the desired dimensions
dimensions = [2, 3, 10, 100, 1000, 10000]

# Generate and print the arrays
distances_list = []
for n in dimensions:
  tensor = generate_n_dimensional_tensor(n)
  num_points = tensor.shape[0]

  distances = []
  for i in range(num_points):
    for j in range(i + 1, num_points):
      distance = euclidean_distance(tensor[i], tensor[j]).item()
      distances.append(distance)
  distances_list.append(distances)

plot_distance_histograms(distances_list, dimensions)

# %%
printlatex("$\\text{Thus we see as }n \\to \\infty, \\text{distances between points increase.}$")

# %%
allDone();
