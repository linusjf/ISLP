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
# # Notes

# %% [markdown]
# ## Import notebook funcs

# %%
from notebookfuncs import *

# %% [markdown]
# ## ROC

# %% [markdown]
# ### What is the ROC Curve?
#
# The ROC curve stands for Receiver Operating Characteristics Curve and is an evaluation metric for classification tasks and it is a probability curve that plots sensitivity and specificity. So, we can say that the ROC Curve can also be defined as the evaluation metric that plots the sensitivity against the false positive rate. The ROC curve plots two different parameters given below:
#
# 1. True positive rate
# 2. False positive rate
#
# The ROC Curve can also defined as a graphical representation that shows the performance or behavior of a classification model at all different threshold levels. The ROC Curve is a tool used for binary classification in machine learning. While learning about the ROC Curve we need to be familiar with the terms specificity and sensitivity.
#
# - Specificity: It is defined as the proportion of negative instances that were predicted correctly as negative values. In other terms, the true negative is also called the specificity. The false positive rate can be found using the specificity by subtracting one from it.
#
# - Sensitivity: The true positive rate is defined as the rate of positive instances that were predicted correctly to be positive. The true positive rate is a synonym for "True positive rate".The sensitivity is also called recall and these terms are often interchangeable. The formula for TPR is as follows,
#
#  TPR = TP / (TP + FN)
#
#  where, TPR = True positive rate, TP = True positive, FN = False negative.
#
#  False positive rate: On the other side, false positive rate can be defined as the rate of negative instances that were predicted incorrectly to be positive. In other terms, the false positive can also be called "1 - specificity".
#
# FPR = FP / (FP + TN)
#
# where, FPR = False positive rate, FP = False positive, TN = True negative.
#
# The ROC Curve is often comparable with the precision and recall curve but it is different because it plots the true positive rate (which is also called recall) against the false positive rate.
#
# The curve is plotted by finding the values of TPR and FPR at distinct threshold values and we don't plot the probabilities but we plot the scores. So the probability of the positive class is taken as the score here.
#
# ### Types of ROC Curve
#
# There are two types of ROC Curves:
#
# 1. Parametric ROC Curve: The parametric method plots the curve using maximum likelihood estimation. This type of ROC Curve is also smooth and plots any sensitivity and specificity, but it has drawbacks like actual data can be discarded. The computation of this method is complex.
#
# 2. Non-Parametric ROC Curve: The non-parametric method does not need any assumptions about the data distributions. It gives unbiased estimates and plot passes through all the data points. The computation of this method is simple.

# %% [markdown]
# ## ROC Curve in Python

# %%
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import roc_curve, auc
from ISLP import load_data
Default = load_data('Default')
Default.columns

# %%
Default.shape

# %%
Default.describe()

# %%
X = Default[["balance", "income"]]

 # %%
 y= Default["default"]

# %%
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# %%
# Initializing the LDA estimator
lda = LinearDiscriminantAnalysis()
# Performing LDA
lda.fit(X, y)
lda.predict(X)
lda.score(X,y)

# %%
# Null error rate
sum(Default["default"] == "No") / len(Default)   # 0.9667

# %%
from sklearn.metrics import confusion_matrix

# Assigning predicted y values
y_pred = lda.predict(X)

# Creating confusion matrix
cm = confusion_matrix(y_true=y, y_pred=y_pred)

# Getting individual values
tn, fp, fn, tp = confusion_matrix(y, y_pred).ravel()
print(tn,fp,fn,tp)

# %%
from sklearn.metrics import ConfusionMatrixDisplay
disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                      display_labels=["No", "Yes"])
disp.plot();

# %%
allDone();
