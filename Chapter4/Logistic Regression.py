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
# # Logistic Regression

# %% [markdown]
# ## Theory and Concepts

# %% [markdown]
# ### What is Logistic Regression, and how does it differ from Linear Regression?

# %% [markdown]
# Logistic Regression and Linear Regression are both supervised learning algorithms used for predicting outcomes, but they differ in the type of outcome they predict and the mathematical approach used:
#
# #### Logistic Regression
#
# Logistic Regression is used for binary classification problems, where the outcome is either 0 or 1, yes or no, etc. It predicts the probability of an event occurring, given a set of input variables.
#
# ##### Key characteristics:
#
# Outcome variable is binary (0/1, yes/no)
#
# Predicts probabilities using a logistic function (sigmoid curve)
#
# Coefficients represent the change in log-odds of the outcome
#
# #### Linear Regression
#
# Linear Regression is used for continuous outcome variables, predicting a numerical value based on one or more input features.
#
# ##### Key characteristics:
#
# Outcome variable is continuous (numeric)
#
# Predicts actual values using a linear equation
#
# Coefficients represent the change in the outcome variable
#
# ##### Key differences:
#
# Outcome variable type: Logistic Regression predicts binary outcomes, while Linear Regression predicts continuous outcomes.
#
# Mathematical approach: Logistic Regression uses a logistic function (sigmoid curve) to predict probabilities, while Linear Regression uses a linear equation to predict actual values.
#
# To illustrate the difference, consider the following examples:
#
# Logistic Regression: Predicting whether a person will buy a car (yes/no) based on their age, income, and credit score.
#
# Linear Regression: Predicting the price of a house based on its size, number of bedrooms, and location.
#
# In summary, Logistic Regression is suitable for binary classification problems, while Linear Regression is suitable for continuous outcome variables.
#
# ##### What is a sigmoid curve?
#
# A sigmoid curve, also known as a logistic curve or S-curve, is a mathematical curve that has an "S" shape. It is a continuous, smooth curve that starts at 0, increases gradually, and then levels off at 1.
#
# Characteristics:
#
# Range: The curve ranges from 0 to 1, making it ideal for modeling probabilities.
#
# Shape: The curve has an "S" shape, where the rate of change is slow at the extremes and rapid in the middle.
#
# Asymptotes: The curve has horizontal asymptotes at 0 and 1, meaning it approaches these values but never reaches them.
#
# Sigmoid function:
#
# The sigmoid function is a mathematical function that maps any real-valued number to a value between 0 and 1. The most common sigmoid function is the logistic function:
# σ(x) = 1 / (1 + e^(-x))
# where e is the base of the natural logarithm (approximately 2.718).
#
# Applications:
#
# Sigmoid curves have numerous applications in:
#
# Logistic Regression: Modeling probabilities of binary outcomes.
#
# Neural Networks: Introducing non-linearity in activation functions.
#
# Probability theory: Modeling cumulative distribution functions.
#
# Biology: Modeling population growth, chemical reactions, and more.
#
# The sigmoid curve's unique shape and properties make it a powerful tool for modeling and analyzing complex phenomena.
#
# ![Sigmoid Curve](Logistic-curve.svg)

# %% [markdown]
# ### What is the logistic function, and how is it used in Logistic Regression?

# %% [markdown]
# #### Logistic Function
#
# The logistic function, also known as the sigmoid function, is a mathematical function that maps any real-valued number to a value between 0 and 1. The logistic function is defined as:
#
# $\sigma(x) = 1 / (1 + e^{-x})$
#
# where:
#
# $\sigma(x)$ is the logistic function
#
# x is the input value
#
# e is the base of the natural logarithm (approximately 2.718)
#
# Logistic Function in Logistic Regression
#
# In Logistic Regression, the logistic function is used to model the probability of a binary outcome (0 or 1, yes or no, etc.) based on one or more input features.
# The logistic function is used to transform the linear combination of input features into a probability value between 0 and 1.
#
# Logistic Regression Equation
#
# The logistic regression equation is:
#
# $p = \sigma(z) = 1 / (1 + e^{-z})$
#
# where:
#
# p is the probability of the positive outcome
#
# z is the linear combination of input features ($w^{T} * x + b$)
#
# w is the weight vector
#
# x is the input feature vector
#
# b is the bias term
#
# The logistic function is used to ensure that the predicted probabilities are between 0 and 1, which is essential for binary classification problems.
#
# Interpretation
#
# The output of the logistic function can be interpreted as:
#
# A probability value close to 0 indicates a low likelihood of the positive outcome.
#
# A probability value close to 1 indicates a high likelihood of the positive outcome.
#
# A probability value close to 0.5 indicates a 50% chance of the positive outcome.
#
# By using the logistic function, Logistic Regression can provide a probabilistic interpretation of the results, making it a powerful tool for binary classification problems.

# %% [markdown]
# ### What is the concept of odds and odds ratio in Logistic Regression?

# %% [markdown]
# In Logistic Regression, odds and odds ratio are essential concepts that help interpret the relationship between the predictor variables and the binary outcome variable.
#
# #### Odds
#
# Odds represent the ratio of the probability of an event occurring to the probability of it not occurring. In the context of Logistic Regression, odds can be calculated as:
#
# Odds = p / (1 - p)
#
# where p is the probability of the positive outcome.
#
# Odds Ratio
#
# The odds ratio (OR) is a measure of the change in odds associated with a one-unit change in a predictor variable, while holding all other predictor variables constant.
#
# The odds ratio can be calculated as:
#
# OR = (Odds of event occurring with predictor) / (Odds of event occurring without predictor)
#
# In Logistic Regression, the odds ratio is calculated using the coefficient (β) of the predictor variable:
#
# $OR = e^{\beta}$
#
# where e is the base of the natural logarithm.
#
# Interpretation of Odds Ratio
#
# The odds ratio has a simple and intuitive interpretation:
#
# OR > 1: The predictor variable increases the odds of the positive outcome.
#
# OR < 1: The predictor variable decreases the odds of the positive outcome.
#
# OR = 1: The predictor variable has no effect on the odds of the positive outcome.
#
# For example, if the odds ratio for a predictor variable is 2.5, it means that a one-unit increase in the predictor variable increases the odds of the positive outcome by 2.5 times.
#
# Example
# Suppose we want to model the probability of a person having a heart attack based on their age and smoking status. The output of the Logistic Regression model might include the following coefficients:
#
# | Predictor | 	Coefficient (β) |	Odds Ratio (OR) |
# | -- | -- | -- |
# | Age |	0.05 |	$e^{0.05} = 1.05$ |
# | Smoking Status (Yes/No)	| 1.2	| $e^{1.2} = 3.32$ |
#
# In this example:
# For every one-year increase in age, the odds of having a heart attack increase by 5% (OR = 1.05).
# Smokers have 3.32 times higher odds of having a heart attack compared to non-smokers (OR = 3.32).
# By examining the odds ratio, we can gain insights into the relationships between the predictor variables and the binary outcome variable.

# %% [markdown]
# ### How does Logistic Regression handle categorical variables?

# %% [markdown]
# Logistic Regression can handle categorical variables through a process called one-hot encoding or dummy coding.
#
# #### One-Hot Encoding
#
# One-hot encoding is a technique where each category of a categorical variable is represented as a binary vector.
# For example, suppose we have a categorical variable "Color" with three categories: Red, Green, and Blue.
#
# | Color	 | One-Hot Encoding |
#  | -- | ------ | 
# | Red	 | 1, 0, 0 |
# | Green |	0, 1, 0 |
# | Blue |	0, 0, 1 |
#
# In this example, each category is represented as a binary vector with three elements. The first element represents Red, the second element represents Green, and the third element represents Blue.
#
# #### Dummy Coding
#
# Dummy coding is similar to one-hot encoding, but it drops one category to avoid multicollinearity.
#
# For example, suppose we have a categorical variable "Color" with three categories: Red, Green, and Blue.
#
# | Color	 | Dummy Coding |
# | -- | ---- |
# | Red |	1, 0 |
# | Green |	0, 1 |
# | Blue |	0, 0  |
#
# In this example, Blue is the reference category, and Red and Green are represented as binary vectors.
#
# Handling Categorical Variables in Logistic Regression
#
# When handling categorical variables in Logistic Regression:
#
# One-hot encoding or dummy coding: Convert the categorical variable into a numerical representation using one-hot encoding or dummy coding.
#
# Create new features: Create new features for each category, except for the reference category (if using dummy coding).
#
# Include new features in the model: Include the new features in the Logistic Regression model.
#
# By handling categorical variables in this way, Logistic Regression can effectively model the relationships between categorical variables and the binary outcome variable.

# %% [markdown]
# #### Are there alternatives to Dummy Coding?

# %% [markdown]
# Yes, there are alternatives to dummy coding for handling categorical variables in Logistic Regression and other machine learning models. Some of these alternatives include:
#
# 1. One-Hot Encoding (OHE)
#
# One-hot encoding is a technique where each category of a categorical variable is represented as a binary vector. This is similar to dummy coding, but it does not drop one category.
#
# 2. Label Encoding
#
# Label encoding is a technique where each category of a categorical variable is assigned a unique integer value. This can be useful when the categorical variable has a natural order or hierarchy.
#
# 3. Ordinal Encoding
#
# Ordinal encoding is a technique where each category of a categorical variable is assigned a unique integer value, taking into account the natural order or hierarchy of the categories.
#
# 4. Hashing Vectorizer
#
# Hashing vectorizer is a technique where each category of a categorical variable is represented as a vector using a hash function. This can be useful when dealing with high-cardinality categorical variables.
#
# 5. Category Embeddings
#
# Category embeddings are a technique where each category of a categorical variable is represented as a dense vector, similar to word embeddings. This can be useful when dealing with high-cardinality categorical variables.
#
# 6. Helmert Coding
#
# Helmert coding is a technique where each category of a categorical variable is compared to the mean of the previous categories. This can be useful when dealing with ordinal categorical variables.
#
# 7. Polynomial Coding
#
# Polynomial coding is a technique where each category of a categorical variable is represented as a polynomial function of the category's integer value. This can be useful when dealing with ordinal categorical variables.
# Each of these alternatives has its own strengths and weaknesses, and the choice of which one to use depends on the specific problem and data.

# %% [markdown]
# #### Provide an example of each except One-hot encoding.

# %% [markdown]
# Here are examples of each of the alternatives to dummy coding, except one-hot encoding:
#
# ##### Label Encoding
#
# Suppose we have a categorical variable "Color" with three categories: Red, Green, and Blue.
#
# | Color	Label | Encoding |
# | -- | ---- |
# | Red |	0 |
# | Green |	1 |
# | Blue |	2 |
#
# In label encoding, each category is assigned a unique integer value.
#
# ##### Ordinal Encoding
#
# Suppose we have a categorical variable "Size" with three categories: Small, Medium, and Large.
#
# | Size |	Ordinal Encoding |
# | -- | ---- |
# | Small |	0 |
# | Medium | 	1 |
# | Large | 2 |
#
# In ordinal encoding, each category is assigned a unique integer value that preserves the natural order or hierarchy of the categories.
#
# ##### Hashing Vectorizer
#
# Suppose we have a categorical variable "Color" with three categories: Red, Green, and Blue.
#
# | Color |	Hashing Vectorizer |
# | -- | ---- |
# | Red	| [0.1, 0.2, 0.3] |
# | Green |	[0.4, 0.5, 0.6] |
# | Blue	| [0.7, 0.8, 0.9] |
#
# In hashing vectorizer, each category is represented as a vector using a hash function.
#
# ##### Category Embeddings
#
# Suppose we have a categorical variable "Color" with three categories: Red, Green, and Blue.
#
# | Color |	Category Embeddings |
# | -- | ---- |
# | Red	| [0.1, 0.2, 0.3, 0.4] |
# | Green |	[0.5, 0.6, 0.7, 0.8] |
# | Blue |	[0.9, 0.1, 0.2, 0.3] |
#
# In category embeddings, each category is represented as a dense vector, similar to word embeddings.
#
# ##### Helmert Coding
#
# Suppose we have a categorical variable "Size" with three categories: Small, Medium, and Large.
#
# | Size |	Helmert Coding |
# | -- | ---- |
# | Small	| [-1, -1] |
# | Medium	| [1, 0] |
# | Large |	[0, 1] |
#
# In Helmert coding, each category is compared to the mean of the previous categories.
#
# ##### Polynomial Coding
# Suppose we have a categorical variable "Size" with three categories: Small, Medium, and Large.
#
# | Size |	Polynomial Coding |
# | -- | ---- |
# | Small	| [1, 0, 0] |
# | Medium |	[1, 1, 0] |
# | Large	| [1, 0, 1] |
#
# In polynomial coding, each category is represented as a polynomial function of the category's integer value.
#
# These examples illustrate how each of these alternatives can be used to represent categorical variables in a machine learning model.

# %% [markdown]
# #### Explain Hashing Vectorizer in detail.

# %% [markdown]
# Hashing Vectorizer is a technique used to transform categorical variables into numerical representations that can be fed into machine learning algorithms.
#
# How Hashing Vectorizer Works
#
# Hashing Vectorizer works by using a hash function to map each category of a categorical variable to a numerical vector. The hash function takes the category as input and generates a fixed-size numerical vector.
#
# Here's a step-by-step overview of the Hashing Vectorizer process:
#
# Category Hashing: Each category of the categorical variable is passed through a hash function, which generates a numerical hash value.
#
# Vector Generation: The hash value is then used to generate a numerical vector of a fixed size (e.g., 128 or 256 dimensions).
#
# Vector Normalization: The generated vector is then normalized to have a length of 1, which helps to prevent feature dominance.
#
# Properties of Hashing Vectorizer
#
# Hashing Vectorizer has several properties that make it useful for transforming categorical variables:
#
# Fixed-size vectors: Hashing Vectorizer generates fixed-size vectors, which makes it easy to integrate with machine learning algorithms.
#
# Efficient computation: Hashing Vectorizer uses a hash function to generate vectors, which makes it computationally efficient.
#
# Handling high-cardinality variables: Hashing Vectorizer can handle categorical variables with a large number of categories (high-cardinality variables).
#
# Advantages of Hashing Vectorizer
#
# Hashing Vectorizer has several advantages that make it a popular choice for transforming categorical variables:
#
# Fast and efficient: Hashing Vectorizer is computationally efficient and can handle large datasets.
#
# Scalable: Hashing Vectorizer can handle high-cardinality variables and large datasets.
#
# Easy to implement: Hashing Vectorizer is easy to implement and integrate with machine learning algorithms.
#
# Disadvantages of Hashing Vectorizer
#
# Hashing Vectorizer also has some disadvantages:
#
# Collision risk: There is a risk of collision, where two different categories hash to the same vector.
#
# Loss of information: Hashing Vectorizer can lose information about the categorical variable, especially if the hash function is not well-designed.
#
# Real-World Applications of Hashing Vectorizer
#
# Hashing Vectorizer has several real-world applications:
#
# Natural Language Processing (NLP): Hashing Vectorizer can be used to transform text data into numerical representations.
#
# Recommendation Systems: Hashing Vectorizer can be used to transform categorical variables into numerical representations for recommendation systems.
#
# Image Classification: Hashing Vectorizer can be used to transform categorical variables into numerical representations for image classification tasks.
#
# In summary, Hashing Vectorizer is a powerful technique for transforming categorical variables into numerical representations. It has several advantages, including fast computation, scalability, and ease of implementation. However, it also has some disadvantages, including collision risk and loss of information.

# %% [markdown]
# ### What is the difference between a logit and a probability?

# %% [markdown]
# ## Model Interpretation and Evaluation
# 1. How do you interpret the coefficients of a Logistic Regression model?
# 2. What is the concept of pseudo-R-squared in Logistic Regression?
# 3. How do you evaluate the performance of a Logistic Regression model?
# 4. What is the difference between accuracy, precision, recall, and F1-score in Logistic Regression?
# 5. How do you handle class imbalance in Logistic Regression?

# %% [markdown]
# ## Model Building and Assumptions
# 1. What are the assumptions of Logistic Regression?
# 2. How do you check for multicollinearity in Logistic Regression?
# 3. What is the concept of interaction terms in Logistic Regression?
# 4. How do you handle missing values in Logistic Regression?
# 5. What is the difference between a simple and a multiple Logistic Regression model?

# %% [markdown]
# ## Applications and Case Studies
# 1. What are some common applications of Logistic Regression?
# 2. How is Logistic Regression used in credit risk assessment?
# 3. What role does Logistic Regression play in medical diagnosis?
# 4. How is Logistic Regression used in marketing and customer churn prediction?
# 5. What are some common challenges faced when implementing Logistic Regression in real-world scenarios?

# %% [markdown]
# ## Advanced Topics
# 1. What is the concept of regularization in Logistic Regression?
# 2. How does L1 and L2 regularization differ in Logistic Regression?
# 3. What is the concept of elastic net regularization in Logistic Regression?
# 4. How does Logistic Regression relate to other machine learning algorithms, such as Decision Trees and Random Forests?
# 5. What are some advanced techniques for handling high-dimensional data in Logistic Regression?
