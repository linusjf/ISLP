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

# %% [raw]
# {{< include codewraplatex.yml >}}

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
import numpy as np
import pandas as pd
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
Default["student"] = Default["student"].astype("category")

# %%
Default = pd.get_dummies(Default,columns=["student"],drop_first=True)
Default["student"] = Default["student_Yes"]

# %%
X = Default[["balance", "income","student"]]

 # %%
 y= Default["default"]

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
### Count of defaults versus non-defaults

# %%
print(sum(Default["default"] == "No"),
sum(Default["default"] == "Yes"))

# %%
from sklearn.metrics import confusion_matrix

# Assigning predicted y values
y_pred = lda.predict(X)

# Creating confusion matrix
cm = confusion_matrix(y_true=y, y_pred=y_pred)
print(cm)

# Getting individual values
tn, fp, fn, tp = confusion_matrix(y, y_pred).ravel()

tn = sum(1 for i, j in zip(y, y_pred) if i == "No" and j == "No")
tp = sum(1 for i, j in zip(y, y_pred) if i == "Yes" and j == "Yes")
fp = sum(1 for i, j in zip(y, y_pred) if i == "No" and j == "Yes")
fn = sum(1 for i, j in zip(y, y_pred) if i == "Yes" and j == "No")
print(tn,fp,fn,tp)

# %%
from sklearn.metrics import ConfusionMatrixDisplay
disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                      display_labels=["No", "Yes"])
disp.plot();

# %% [markdown]
# ### Accuracy

# %% [markdown]
# Accuracy is defined as
#
# (number of true negatives + number of true positives) /
# (number of true negatives + number of true positives + number of false negatives +
# number of false positives)

# %%
Accuracy = (tn + tp) / (tn + tp + fn + fp)

# %% [markdown]
# ### Precision
#
# Precision is defined as
#
# (number of true positives) / (number of true positives + number of false positives)
#
# = TP / P*
#
# where P* is the total number of predicted positives (defaults) in the dataset.

# %%
Precision = tp / (tp + fp)

# %% [markdown]
# Intuitively, precision is representing the proportion of all our predicted positive values that are actually positive. When comparing different classification models, precision is a good measure when we want to avoid false positives.
#
# For example, when detecting spam emails, a model with high precision is likely preferred. In the case of spam email, the email user would much rather get the occasional spam email (a false negative) than miss an important email that wasn't spam (a false positive).

# %% [markdown]
# ### Recall

# %% [markdown]
# Recall is defined as
#
# (number of true positives) / (number of true positives + number of false negatives)
# = TP / P
#
# where P is actual positives (defaults) in the dataset.

# %%
Recall = tp / (tp + fn)

# %% [markdown]
# Intuitively, recall is representing how well a model is classifying positive observations as actually positive. When comparing different classification models, recall is a good measure when we want to avoid false negatives.

# %% [markdown]
# Recall is a metric of particular interest. When classifying events as anomalous or not, we would much rather classify a non-anomalous event as anomalous (a false positive), than misclassify an actual anomaly as non-anomalous (a false negative). Said another way, out of all the actual anomalies out there, we want to make sure we detect as many as we can, even at the expense of including some false positives.
#
# In the medical setting, recall is more commonly referred to as sensitivity. A related term in the medical literature is specificity, which is equivalent to the true negative rate. Occasionally, specificity is also referred to as recall of the negative class.

# %% [markdown]
# ### $F_1$ score

# %% [markdown]
# The $F_1$ score is the harmonic mean of precision and recall, and is defined as
#
# $$\large 2 * \frac { precision \: * \: recall} {precision \: + \: recall}$$
#
# When comparing models, the $\large F_1$ score is useful when we want to strike a balance between precision and recall. That is, when we want to avoid both false positives (as in spam email classification) and false negatives (as in anomaly detection).
#

# %%
F1 = 2 * Precision * Recall / (Precision + Recall)

# %% [markdown]
# ### scikit-learn's Classification Report

# %%
from sklearn.metrics import classification_report
print(classification_report(y_true=y,
                                y_pred=y_pred,
                                target_names=["No", "Yes"]))

# %% [markdown]
# Note that our manually calculated values of precision, recall, and the $F_1$
#  score align with the Yes row above. For completeness, support simply refers to the number of observations in each class. For our working example, that is number of observations that did not default (9667) or that did default (333) present in our training data.

# %% [markdown]
# ## Train Logistic Regression Model

# %%
from sklearn.linear_model import LogisticRegression

# Setting penalty=None to disable regularization
log_reg = LogisticRegression(penalty=None)
log_reg.fit(X, y)
log_reg.score(X, y)  # 0.9732

# %%
from sklearn.metrics import RocCurveDisplay
lr_disp = RocCurveDisplay.from_estimator(log_reg,X,y,alpha=0.8,plot_chance_level=True)
RocCurveDisplay.from_estimator(lda,X,y,ax=lr_disp.ax_,alpha=0.8)
plt.title("ROC Curve Comparison")
plt.show()

# %% [markdown]
# ## References:
# 1. [Linear discriminant analysis #2
# scikit-learn, precision, recall, F-scores, ROC curves, and a comparison to logistic regression](https://ethanwicker.com/2021-02-07-linear-discriminant-analysis-002/)

# %%
allDone();
