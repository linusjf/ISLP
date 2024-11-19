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
from ISLP import confusion_table
from ISLP.models import (ModelSpec as MS , summarize)
from summarytools import dfSummary
import numpy as np
import klib
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.tools.tools import add_constant
import pandas as pd
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import ConfusionMatrixDisplay

# %% [markdown]
# ## Exercise 13

# %% [markdown]
# This question should be answered using the Weekly data set, which is part of the ISLP package. This data is similar in nature to the Smarket data from this chapter's lab, except that it contains 1,089 weekly returns for 21 years, from the beginning of 1990 to the end of 2010.

# %%
Weekly = load_data("Weekly")
Weekly["LogVolume"] = np.log(Weekly["Volume"])
Weekly = Weekly.drop(columns=["Volume"])
Weekly = klib.convert_datatypes(Weekly)
print(Weekly.dtypes)
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

# %%
# Try to avoid using ISLP classes for anything else but to load data since it may not transfer well to
# actual usage in data analysis projects
# drop columns Today, Direction, Year
allvars = Weekly[Weekly.columns.difference(['Today', 'Direction', 'Year'])]
# add constant term of 1s
X = add_constant(allvars)
# Convert 'Down' and 'Up' to 0s and 1s respectively
y = Weekly.Direction == 'Up'
# Use Binomial family for Logistic Regression
family = sm.families.Binomial()
glm = sm.GLM(y, X, family=family)
results = glm.fit()
summarize(results)

# %%
results.summary()

# %%
results.model.endog_names

# %%
results.params

# %%
results.pvalues[results.pvalues < 0.05]

# %% [markdown]
# From the above, it can be deduced that `Lag2` is the only significant variable that predicts `Direction`. The positive coefficient for `Lag2` suggests that if the market had a positive return today, it is more likely that the market will rise once more in two days and vice versa. We can also see that the confidence intervals of the other parameters `Lag1, Lag3, Lag4, Lag5 and LogVolume` span the value 0 and thus are not significant.

# %% [markdown]
# ## (c)

# %% [markdown]
# Compute the confusion matrix and overall fraction of correct predictions. Explain what the confusion matrix is telling you about the types of mistakes made by logistic regression.

# %%
predictions = results.predict()

# %%
labels = np.array(['Down']*len(Weekly))
labels[predictions > 0.5] = "Up"

# %%
ct = confusion_table(Weekly["Direction"],
                       labels)
ct

# %% [markdown]
# The diagonal elements of the confusion matrix indicate correct predictions, while the off-diagonals represent incorrect predictions. Hence our model correctly predicted that the market would go up on 553 days and that it would go down on 59 days, for a total of 553 + 59 = 612 correct predictions. The `np.mean()` function can be used to compute the fraction of days for which the prediction was correct. In this case, logistic regression correctly predicted the movement of the market 56.2% of the time.

# %%
np.mean(labels == Weekly.Direction), (553+59) /len(Weekly)

# %% [markdown]
# This accuracy of 56.2% is not much better than the no information classifier's (NIC)  accuracy of 55.56% when we just guess that the market will go up all the time and achieve an accuracy level of 55.56%.

# %% [markdown]
# 100 - 56.2 = 43.8% is the training error rate. As we have seen previously, the training error rate is often overly optimistic &mdash; it tends to underestimate the test error rate. In order to better assess the accuracy of the logistic regression model in this setting, we can fit the model using part of the data, and then examine how well it predicts the held out data. This will yield a more realistic error rate, in the sense that in practice we will be interested in our model's performance not on the data that we used to fit the model, but rather on days in the future for which the market's movements are unknown.

# %%
print(classification_report(Weekly["Direction"],
                            labels,
                            digits = 3))
cm = confusion_matrix(Weekly["Direction"],
                       labels)

disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                      display_labels=["Down", "Up"])
disp.plot();

# %%
# Getting individual values for
true_negatives, false_positives, false_negatives, true_positives = cm.ravel()

# %%
support_up = np.sum(Weekly["Direction"] == "Up")
support_down = Weekly.shape[0] - support_up
predicted_up = np.sum(labels == "Up")
predicted_down = np.sum(labels == "Down")
predicted_correctly_up = true_positives
precision_up = predicted_correctly_up / predicted_up
predicted_correctly_down = true_negatives
precision_down = predicted_correctly_down / predicted_down
print(f"Support (Up, Down): {support_up}, {support_down}")
print(f"Precision (Up, Down): {precision_up:.3f}, {precision_down:.3f}")
print(f"Precision Average (Macro, Weighted): {(precision_up + precision_down)/2:.3f}, {(precision_up * support_up + precision_down * support_down)/(support_up + support_down):.3f}")
recall_up = true_positives / support_up
recall_down = true_negatives / support_down
print(f"Recall (Up, Down): {recall_up:.3f}, {recall_down:.3f}")
print(f"Recall Average (Macro, Weighted): {(recall_up + recall_down)/2:.3f},{(recall_up * support_up + recall_down * support_down)/(support_up + support_down):.3f}")
f1_score_up = 2 * precision_up * recall_up/ (precision_up + recall_up)
f1_score_down = 2 * precision_down * recall_down/ (precision_down + recall_down)
print(f"F1 score (Up, Down): {f1_score_up:.3f}, {f1_score_down:.3f}")
print(f"F1 score Average (Macro, Weighted): {(f1_score_up + f1_score_down)/2:.3f},{(f1_score_up * support_up + f1_score_down * support_down)/(support_up + support_down):.3f}")

# %% [markdown]
# ### (d)

# %% [markdown]
# Now fit the logistic regression model using a training data period from 1990 to 2008, with Lag2 as the only predictor. Compute the confusion matrix and the overall fraction of correct predictions for the held out data (that is, the data from 2009 and 2010).

# %%
train = (Weekly.Year <= 2008)
Weekly_train = Weekly.loc[train]
Weekly_test = Weekly.loc[~train];

# %%
Weekly_train.shape

# %%
Weekly_test.shape

# %%
X_train , X_test = Weekly_train["Lag2"], Weekly_test["Lag2"]
# add constant term of 1s
X_train = add_constant(X_train)
X_test = add_constant(X_test)
y_train , y_test = Weekly_train["Direction"] == "Up", Weekly_test["Direction"] == "Up"
glm_train = sm.GLM(y_train , X_train, family=sm.families.Binomial())
results = glm_train.fit()
probs = results.predict(exog=X_test);

# %%
D = Weekly.Direction
L_train , L_test = D.loc[train], D.loc[~train]

# %%
labels = np.array(['Down']*L_test.shape[0])
labels[probs > 0.5] = 'Up'
confusion_table(labels , L_test)

# %%
np.mean(labels == L_test), np.mean(labels != L_test)

# %%
print(classification_report(Weekly_test["Direction"],
                            labels,
                            digits = 3))
cm = confusion_matrix(Weekly_test["Direction"],
                       labels)
disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                      display_labels=["Down", "Up"])
disp.plot();

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
