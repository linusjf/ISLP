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
# # What is Logistic Regression, and how does it differ from Linear Regression?
#
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
# **Outcome variable type**: Logistic Regression predicts binary outcomes, while Linear Regression predicts continuous outcomes.
#
# **Mathematical approach**: Logistic Regression uses a logistic function (sigmoid curve) to predict probabilities, while Linear Regression uses a linear equation to predict actual values.
#
# To illustrate the difference, consider the following examples:
#
# **Logistic Regression**: Predicting whether a person will buy a car (yes/no) based on their age, income, and credit score.
#
# **Linear Regression**: Predicting the price of a house based on its size, number of bedrooms, and location.
#
# In summary, Logistic Regression is suitable for binary classification problems, while Linear Regression is suitable for continuous outcome variables.
#
# ##### What is a sigmoid curve?
#
# A sigmoid curve, also known as a logistic curve or S-curve, is a mathematical curve that has an "S" shape. It is a continuous, smooth curve that starts at 0, increases gradually, and then levels off at 1.
#
# **Characteristics**:
#
# **Range**: The curve ranges from 0 to 1, making it ideal for modeling probabilities.
#
# **Shape**: The curve has an "S" shape, where the rate of change is slow at the extremes and rapid in the middle.
#
# **Asymptotes**: The curve has horizontal asymptotes at 0 and 1, meaning it approaches these values but never reaches them.
#
# **Sigmoid function:**
#
# The sigmoid function is a mathematical function that maps any real-valued number to a value between 0 and 1. The most common sigmoid function is the logistic function:
# $\sigma(x) = 1 / (1 + e^{-x})$
# where e is the base of the natural logarithm (approximately 2.718).
#
# **Applications:**
#
# Sigmoid curves have numerous applications in:
#
# **Logistic Regression:** Modeling probabilities of binary outcomes.
#
# **Neural Networks:** Introducing non-linearity in activation functions.
#
# **Probability theory:** Modeling cumulative distribution functions.
#
# **Biology:** Modeling population growth, chemical reactions, and more.
#
# The sigmoid curve's unique shape and properties make it a powerful tool for modeling and analyzing complex phenomena.
#
# ![Sigmoid Curve](Logistic-curve.png)
#
# ### What is the logistic function, and how is it used in Logistic Regression?
#
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
# **Logistic Regression Equation**
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
# **Interpretation**
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
#
# ### What is the concept of odds and odds ratio in Logistic Regression?
#
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
# **Odds Ratio**
#
# The odds ratio (OR) is a measure of the change in odds associated with a one-unit change in a predictor variable, while holding all other predictor variables constant.
#
# The odds ratio can be calculated as:
#
# OR = (Odds of event occurring with predictor) / (Odds of event occurring without predictor)
#
# In Logistic Regression, the odds ratio is calculated using the coefficient $(\beta)$ of the predictor variable:
#
# $OR = e^{\beta}$
#
# where e is the base of the natural logarithm.
#
# **Interpretation of Odds Ratio**
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
#
# ### How does Logistic Regression handle categorical variables?
#
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
# **Handling Categorical Variables in Logistic Regression**
#
# When handling categorical variables in Logistic Regression:
#
# One-hot encoding or dummy coding: Convert the categorical variable into a numerical representation using one-hot encoding or dummy coding.
#
# **Create new features:** Create new features for each category, except for the reference category (if using dummy coding).
#
# Include new features in the model: Include the new features in the Logistic Regression model.
#
# By handling categorical variables in this way, Logistic Regression can effectively model the relationships between categorical variables and the binary outcome variable.
#
# #### Are there alternatives to Dummy Coding?
#
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
#
# #### Provide an example of each except One-hot encoding.
#
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
#
# #### Explain Hashing Vectorizer in detail.
#
# Hashing Vectorizer is a technique used to transform categorical variables into numerical representations that can be fed into machine learning algorithms.
#
# **How Hashing Vectorizer Works**
#
# Hashing Vectorizer works by using a hash function to map each category of a categorical variable to a numerical vector. The hash function takes the category as input and generates a fixed-size numerical vector.
#
# Here's a step-by-step overview of the Hashing Vectorizer process:
#
# **Category Hashing**: Each category of the categorical variable is passed through a hash function, which generates a numerical hash value.
#
# **Vector Generation**: The hash value is then used to generate a numerical vector of a fixed size (e.g., 128 or 256 dimensions).
#
# **Vector Normalization**: The generated vector is then normalized to have a length of 1, which helps to prevent feature dominance.
#
# **Properties of Hashing Vectorizer**
#
# Hashing Vectorizer has several properties that make it useful for transforming categorical variables:
#
# **Fixed-size vectors**: Hashing Vectorizer generates fixed-size vectors, which makes it easy to integrate with machine learning algorithms.
#
# **Efficient computation**: Hashing Vectorizer uses a hash function to generate vectors, which makes it computationally efficient.
#
# **Handling high-cardinality variables:** Hashing Vectorizer can handle categorical variables with a large number of categories (high-cardinality variables).
#
# **Advantages of Hashing Vectorizer**
#
# Hashing Vectorizer has several advantages that make it a popular choice for transforming categorical variables:
#
# **Fast and efficient:** Hashing Vectorizer is computationally efficient and can handle large datasets.
#
# **Scalable:** Hashing Vectorizer can handle high-cardinality variables and large datasets.
#
# **Easy to implement:** Hashing Vectorizer is easy to implement and integrate with machine learning algorithms.
#
# **Disadvantages of Hashing Vectorizer**
#
# Hashing Vectorizer also has some disadvantages:
#
# **Collision risk:** There is a risk of collision, where two different categories hash to the same vector.
#
# Loss of information: Hashing Vectorizer can lose information about the categorical variable, especially if the hash function is not well-designed.
#
# **Real-World Applications of Hashing Vectorizer**
#
# **Hashing Vectorizer has several real-world applications:**
#
# **Natural Language Processing (NLP):** Hashing Vectorizer can be used to transform text data into numerical representations.
#
# **Recommendation Systems:** Hashing Vectorizer can be used to transform categorical variables into numerical representations for recommendation systems.
#
# **Image Classification:** Hashing Vectorizer can be used to transform categorical variables into numerical representations for image classification tasks.
#
# In summary, Hashing Vectorizer is a powerful technique for transforming categorical variables into numerical representations. It has several advantages, including fast computation, scalability, and ease of implementation. However, it also has some disadvantages, including collision risk and loss of information.
#
# ##### Provide python example code for Hashing Vectorizer
#
#
# # Here's an example of using HashingVectorizer with categorical data:
#
# from sklearn.feature_extraction.text import HashingVectorizer
# import numpy as np
# import pandas as pd
#
# # Create a sample dataframe with categorical data
# data = pd.DataFrame({
#     'Color': ['Red', 'Green', 'Blue', 'Red', 'Green', 'Blue'],
#     'Size': ['Small', 'Medium', 'Large', 'Small', 'Medium', 'Large']
# })
#
# # Create a HashingVectorizer object
# vectorizer = HashingVectorizer(n_features=10)
#
# # Fit and transform the data
# X_color = vectorizer.fit_transform(data['Color'])
# X_size = vectorizer.fit_transform(data['Size'])
#
# # Print the transformed data
# print("Color Features:")
# print(X_color.toarray())
# print("\nSize Features:")
# print(X_size.toarray())
#
# In this example:
#
# We create a sample dataframe data with two categorical columns: `Color` and `Size`.
#
# We create a `HashingVectorizer` object with `n_features=10`.
#
# We fit and transform the `Color` and `Size` columns separately using the `fit_transform` method.
#
# We print the transformed data as `NumPy` arrays.
#
# The output will show the transformed categorical data as numerical features. Note that the actual output will depend on the hash function used by the `HashingVectorizer`, so your output may differ from the example output.
#
# This example demonstrates how `HashingVectorizer` can be used to transform categorical data into numerical features that can be used in machine learning models.
#
# ##### What are category embeddings? What are word embeddings? How do they differ from each other?
#
# Category embeddings and word embeddings are both techniques used in natural language processing (NLP) and machine learning to represent categorical variables and words as numerical vectors.
#
# **Category Embeddings**
#   
# Category embeddings are a technique used to represent categorical variables as numerical vectors. The idea is to map each category to a unique vector in a high-dimensional space, such that categories with similar properties or meanings are closer together.
#
# Category embeddings can be learned using various techniques, such as:
#
# Neural networks
#
# Matrix factorization
#
# Word2Vec (adapted for categorical variables)
#
# Category embeddings are useful when dealing with high-cardinality categorical variables, where traditional encoding methods like one-hot encoding may not be effective.
#
# **Word Embeddings**
#
# Word embeddings are a technique used to represent words as numerical vectors. The idea is to map each word to a unique vector in a high-dimensional space, such that words with similar meanings or contexts are closer together.
#
# Word embeddings can be learned using various techniques, such as:
#
# Word2Vec
#
# GloVe
#
# FastText
#
# Word embeddings are useful in NLP tasks like text classification, sentiment analysis, and language modeling.
#
# **Key Differences**
#
# Here are the key differences between category embeddings and word embeddings:
#
# Input data: Category embeddings are used for categorical variables, while word embeddings are used for words or text data.
#
# Learning objectives: Category embeddings aim to capture the relationships between categories, while word embeddings aim to capture the semantic relationships between words.
#
# Vector space: Category embeddings typically exist in a lower-dimensional vector space compared to word embeddings.
#
# Interpretability: Word embeddings are often more interpretable than category embeddings, as the vector dimensions can be related to specific semantic features.
#
# In summary, category embeddings and word embeddings are both used to represent data as numerical vectors, but they differ in their input data, learning objectives, vector space, and interpretability.
#
# ##### What are high-cardinality variables?
#
# High-cardinality variables are categorical variables that have a large number of unique categories or levels. In other words, they are variables that can take on many different values.
#
# **Characteristics of High-Cardinality Variables**
#   
# High-cardinality variables typically have the following characteristics:
#   
# Large number of unique values: High-cardinality variables have many unique categories or levels, often exceeding hundreds or thousands.
#   
# Low frequency of each value: Each unique value in a high-cardinality variable may appear only a few times in the dataset.
#   
# High dimensionality: High-cardinality variables can lead to high-dimensional data, making it challenging to analyze and model.
#   
# **Examples of High-Cardinality Variables**
#   
# Here are some examples of high-cardinality variables:
#   
# Product IDs: In an e-commerce dataset, the product ID variable can have thousands or millions of unique values.
#
# Customer IDs: In a customer database, the customer ID variable can have millions of unique values.
#
# Geographic locations: In a dataset with geographic locations, the city or zip code variable can have thousands of unique values.
#
# Text data: In a text dataset, the words or phrases variable can have tens of thousands of unique values.
#
# **Challenges with High-Cardinality Variables**
#
# High-cardinality variables can pose several challenges:
#
# Dimensionality curse: High-cardinality variables can lead to high-dimensional data, making it difficult to analyze and model.
#
# Overfitting: Models can overfit to the unique values in high-cardinality variables, leading to poor generalization performance.
#
# Computational complexity: High-cardinality variables can increase computational complexity, making it challenging to train models.
#
# **Techniques for Handling High-Cardinality Variables**
#
# Several techniques can be used to handle high-cardinality variables:
#
# Dimensionality reduction: Techniques like PCA, t-SNE, or autoencoders can reduce the dimensionality of high-cardinality variables.
#
# Feature hashing: Feature hashing can be used to reduce the dimensionality of high-cardinality variables by hashing the unique values into a fixed-size vector.
#
# Embeddings: Embeddings can be used to represent high-cardinality variables as dense vectors, reducing the dimensionality and improving model performance.
#
# Grouping or binning: Grouping or binning can be used to reduce the number of unique values in high-cardinality variables.
#
# ##### Explain Helmert Coding in detail.
#
# With Helmert coding, each level of the variable is compared to "later" levels of the variable.
#
# The weights depend on the number of levels of the variable.
#
# If there are L levels then the first comparison is of level vs. (L−1) other levels. The weights are then (L−1)/L for the first level and −1/L for each of the other levels.
#
# The next comparison has only L−1 levels (the first level is no longer part of the comparisons), so now the weights are (L−2)/(L−1) for the first level and −1/(L−1) for the others. And so on.
#
# ![Helmert Coding Example](cyaJM.png)
#
# Helmert coding and its inverse, difference coding, really only make sense when the variable is ordinal.
#
# This system is useful when the levels of the categorical variable are ordered in a meaningful way. For example, if we had a categorical variable in which work-related stress was coded as low, medium or high, then comparing the means of the previous levels of the variable would make more sense.
#
# References:
#
# 1. [How to calculate Helmert encoding](https://stats.stackexchange.com/questions/411134/how-to-calculate-helmert-coding)
#
# **Properties of Helmert Coding**
#
# Helmert coding has several properties that make it useful:
#
# Orthogonality: The resulting variables are orthogonal to each other, which means that they are uncorrelated.
#
# Comparisons: Helmert coding allows for comparisons between each level of the categorical variable and the mean of the previous levels.
#
# Reduced dimensionality: Helmert coding can reduce the dimensionality of the categorical variable, making it easier to analyze and model.
#
# **Advantages of Helmert Coding**
#
# Helmert coding has several advantages:
#
# Easy to interpret: The resulting variables are easy to interpret, as they represent comparisons between each level of the categorical variable and the mean of the previous levels.
#
# Reduced multicollinearity: Helmert coding can reduce multicollinearity between the levels of the categorical variable.
#
# Improved model performance: Helmert coding can improve the performance of machine learning models by reducing the dimensionality of the categorical variable.
#
# **Disadvantages of Helmert Coding**
#
# Helmert coding also has some disadvantages:
#
# Assumes ordinality: Helmert coding assumes that the levels of the categorical variable have an ordinal relationship, which may not always be the case.
#
# Can be sensitive to ordering: The resulting variables can be sensitive to the ordering of the levels of the categorical variable.
#
# **Real-World Applications of Helmert Coding**
#
# Helmert coding has several real-world applications:
#
# Marketing research: Helmert coding can be used to analyze the effect of different marketing campaigns on customer behavior.
#
# Medical research: Helmert coding can be used to analyze the effect of different treatments on patient outcomes.
#
# Social sciences: Helmert coding can be used to analyze the effect of different social factors on behavior and outcomes.
#
# ##### Explain polynomial encoding in detail.
#
# Polynomial coding, also known as polynomial contrasts or orthogonal polynomial coding, is a technique used in statistics and machine learning to encode categorical variables with ordered levels.
#
# **How Polynomial Coding Works**
#
# Polynomial coding works by creating a set of orthogonal contrasts that capture the polynomial relationships between the levels of the categorical variable. The resulting codes are a set of orthogonal vectors that can be used as inputs to a statistical model or machine learning algorithm.
#
# **Key Properties of Polynomial Coding**
#
# Polynomial coding has several key properties:
#
# Orthogonality: The resulting codes are orthogonal to each other, which means that they are uncorrelated.
#
# Polynomial relationships: The codes capture the polynomial relationships between the levels of the categorical variable.
#
# Order preservation: The codes preserve the order of the levels, which means that the resulting codes reflect the underlying order of the categorical variable.
#
# **Types of Polynomial Coding**
#
# There are several types of polynomial coding, including:
#
# Linear coding: Captures the linear relationship between the levels.
#
# Quadratic coding: Captures the quadratic relationship between the levels.
#
# Cubic coding: Captures the cubic relationship between the levels.
#
# **Advantages of Polynomial Coding**
#
# Polynomial coding has several advantages:
#
# Captures complex relationships: Polynomial coding can capture complex polynomial relationships between the levels of the categorical variable.
#
# Preserves order: Polynomial coding preserves the order of the levels, which is important in many applications.
#
# Orthogonal codes: The resulting codes are orthogonal, which makes them easy to interpret and use in statistical models and machine learning algorithms.
#
# **Disadvantages of Polynomial Coding**
#
# Polynomial coding also has some disadvantages:
#
# Can be sensitive to scaling: Polynomial coding can be sensitive to the scaling of the categorical variable.
#
# Can be difficult to interpret: The resulting codes can be difficult to interpret, especially for high-degree polynomials.
#
# **Real-World Applications of Polynomial Coding**
#
# Polynomial coding has many real-world applications:
#
# Dose-response modeling: Polynomial coding can be used to model the dose-response relationship in pharmaceutical applications.
#
# Marketing research: Polynomial coding can be used to model the relationship between marketing variables and customer behavior.
#
# Social sciences: Polynomial coding can be used to model the relationship between social variables and outcomes.
#
# ##### Are there any python libraries that perform polynomial encoding for categories?
#
# Yes, there are several Python libraries that can perform polynomial encoding for categorical variables:
#
# Category Encoders: This library provides a range of encoding methods, including polynomial encoding. You can use the `PolynomialEncoder` class to perform polynomial encoding.
#
# Scikit-learn: While `scikit-learn` doesn't have a specific polynomial encoding method, you can use the `OrdinalEncoder` class to encode categorical variables as integers, and then use the `PolynomialFeatures` class from the `sklearn.preprocessing` module to generate polynomial features.
#
# Pandas: Pandas doesn't have a built-in polynomial encoding method, but you can use the Categorical data type to encode categorical variables, and then use the numpy library to generate polynomial features.

# %%
import category_encoders as ce
import pandas as pd

# Create a sample dataframe
data = {'Category': ['A', 'B', 'C', 'A', 'B', 'C']}
df = pd.DataFrame(data)

# Perform polynomial encoding
encoder = ce.PolynomialEncoder()
encoded_df = encoder.fit_transform(df)

print(encoded_df)
# And here's an example of how to use scikit-learn to perform polynomial encoding:

from sklearn.preprocessing import OrdinalEncoder, PolynomialFeatures
import pandas as pd
import numpy as np

# Create a sample dataframe
data = {'Category': ['A', 'B', 'C', 'A', 'B', 'C']}
df = pd.DataFrame(data)

# Encode categorical variable as integers
ordinal_encoder = OrdinalEncoder()
encoded_categories = ordinal_encoder.fit_transform(df['Category'].values.reshape(-1, 1))

# Generate polynomial features
poly_features = PolynomialFeatures(degree=2)
polynomial_encoded = poly_features.fit_transform(encoded_categories)

print(polynomial_encoded)

# %% [markdown]
# ##### Can you plot example curves for linear, quadratic and cubic polynomial codings?

# %%
# Here's an example code that plots the curves for linear, quadratic, and cubic polynomial codings:

import numpy as np
import matplotlib.pyplot as plt

# Define the number of levels
k = 4

# Define the levels (e.g., dosage levels)
levels = np.arange(1, k + 1)

# Compute linear coding
linear_coding = np.array([-3/2, -1/2, 1/2, 3/2])

# Compute quadratic coding
quadratic_coding = np.array([1, -1, -1, 1])

# Compute cubic coding
cubic_coding = np.array([-1, 3, -3, 1])

# Create a range of x values for plotting
x = np.linspace(1, k, 100)

# Plot the curves
plt.figure(figsize=(10, 6))

plt.subplot(1, 3, 1)
plt.plot(x, (x - (k + 1) / 2) * 3 / (k - 1), label='Linear')
plt.title('Linear Polynomial Coding')
plt.xlabel('Level')
plt.ylabel('Coding Value')
plt.legend()

plt.subplot(1, 3, 2)
plt.plot(x, 3 * (x - (k + 1) / 2) ** 2 - (k ** 2 - 1) / 4, label='Quadratic')
plt.title('Quadratic Polynomial Coding')
plt.xlabel('Level')
plt.ylabel('Coding Value')
plt.legend()

plt.subplot(1, 3, 3)
plt.plot(x, 5 * (x - (k + 1) / 2) ** 3 - 3 * (x - (k + 1) / 2) * (k ** 2 - 1) / 4, label='Cubic')
plt.title('Cubic Polynomial Coding')
plt.xlabel('Level')
plt.ylabel('Coding Value')
plt.legend()

plt.tight_layout()
plt.show()

# %% [markdown]
# This code will generate three plots, one for each type of polynomial coding. Each plot shows the curve of the polynomial coding values as a function of the level.
# Note that the x-axis represents the levels, and the y-axis represents the corresponding polynomial coding values. The curves are plotted using a range of x values to provide a smooth representation of the polynomial coding curves.
#
# ##### Flowchart for Categorical Encoding
#
# ![Categorical Encoding Flowchart](categoricalencodingflowchart.png)
#
# #### Further Reading for Categorical Encoding
#
# 1. <https://github.com/alteryx/categorical_encoding/blob/main/guides/notebooks/categorical-encoding-guide.ipynb>
# 2. <https://www.kaggle.com/code/ptfrwrd/different-approaches-for-categorical-encoding>
# 3. <https://www.linkedin.com/pulse/encode-categorical-features-revanth-yadama/>
# 4. <https://www.kaggle.com/code/discdiver/category-encoders-examples>
# 5. <https://www.kaggle.com/code/paulrohan2020/tutorial-encoding-categorical-variables>
#
# ### What is the difference between a logit and a probability?
#
# A logit and a probability are two related but distinct concepts in statistics and machine learning:
#
# **Probability**
#
# A probability is a value between 0 and 1 that represents the likelihood or chance of an event occurring. Probabilities are often denoted as P(event) or p.
#
# Example: The probability of flipping a coin and getting heads is 0.5.
#
# **Logit**
#
# A logit, also known as a log-odds, is the logarithm of the odds of an event occurring. Logits are often denoted as log(p/(1-p)).
#
# Example: The logit of the probability 0.5 is log(0.5/(1-0.5)) = 0.
#
# **Key differences**:
#
# Range: Probabilities range from 0 to 1, while logits range from negative infinity to positive infinity.
#
# Interpretation: Probabilities represent the likelihood of an event, while logits represent the logarithm of the odds of an event.
#
# Relationship: The logit function maps probabilities to logits, and the inverse logit function (also known as the sigmoid function) maps logits back to probabilities.
#
# The logit function is often used in logistic regression and other machine learning models to model binary outcomes, as it provides a convenient and interpretable way to represent probabilities.
#
# ## Model Interpretation and Evaluation
#
# ### How do you interpret the coefficients of a Logistic Regression model?
#
# **Coefficient Interpretation**
#
# In Logistic Regression, the coefficients represent the change in the log-odds of the outcome variable for a one-unit change in the predictor variable, while holding all other predictor variables constant.
#
# **Log-Odds**
#
# The log-odds are the logarithm of the odds of the outcome variable. The odds are the ratio of the probability of the outcome variable being 1 (e.g., success, yes, etc.) to the probability of it being 0 (e.g., failure, no, etc.).
#
# **Coefficient Sign**
#
# Positive coefficient: Indicates that an increase in the predictor variable is associated with an increase in the log-odds of the outcome variable. This means that the predictor variable is positively related to the outcome variable.
#
# Negative coefficient: Indicates that an increase in the predictor variable is associated with a decrease in the log-odds of the outcome variable. This means that the predictor variable is negatively related to the outcome variable.
#
# **Coefficient Magnitude**
#
# The magnitude of the coefficient represents the strength of the relationship between the predictor variable and the outcome variable.
#
# Large coefficient: Indicates a strong relationship between the predictor variable and the outcome variable.
#
# Small coefficient: Indicates a weak relationship between the predictor variable and the outcome variable.
#
# Example
#
# Suppose we have a Logistic Regression model that predicts the probability of a person being admitted to a university based on their GPA.
#
# | Predictor |	Coefficient |
# | :- | -: |
# | GPA |	2.5 |
#
# The coefficient for GPA is 2.5, which is positive. This means that an increase in GPA is associated with an increase in the log-odds of being admitted to the university.
#
# The magnitude of the coefficient is 2.5, which indicates a strong relationship between GPA and the probability of being admitted.
#
# To make the interpretation more concrete, we can calculate the odds ratio (OR) for a one-unit change in GPA:
#
# $OR = e^{2.5} ≈ 12.18$
#
# This means that for every one-unit increase in GPA, the odds of being admitted to the university increase by a factor of approximately 12.18.
#
# ### What is the concept of pseudo-R-squared in Logistic Regression?
#
# **Pseudo-R-squared** is a measure of goodness of fit for Logistic Regression models, which is analogous to the R-squared measure used in linear regression. However, unlike R-squared, pseudo-R-squared is not a direct measure of the proportion of variance explained by the model.
#
# **Why do we need pseudo-R-squared?**
#
# In Logistic Regression, the response variable is binary (0/1, yes/no, etc.), and the model estimates probabilities rather than continuous values. As a result, the traditional R-squared measure is not applicable.
#
# Types of pseudo-R-squared
#
# There are several types of pseudo-R-squared measures, each with its own strengths and limitations:
#
# *Cox-Snell R-squared*: This measure is based on the likelihood ratio statistic and can be interpreted similarly to traditional R-squared.
#
# *Nagelkerke R-squared*: This measure is an extension of the Cox-Snell R-squared and provides a more accurate estimate of the model's explanatory power.
#
# *McFadden R-squared*: This measure is based on the likelihood ratio statistic and is often used in econometrics.
#
# *Tjur R-squared*: This measure is based on the difference between the predicted probabilities and the actual outcomes.
#
# **Interpretation of pseudo-R-squared**
#
# Pseudo-R-squared values range from 0 to 1, where:
#
# 0: The model does not explain any variation in the response variable.
#
# 1: The model explains all variation in the response variable.
#
# In general, pseudo-R-squared values can be interpreted as follows:
#
# 0.2 or less: Weak model fit
#
# 0.2-0.4: Moderate model fit
#
# 0.4-0.6: Strong model fit
#
# 0.6 or higher: Excellent model fit
#
# Keep in mind that pseudo-R-squared values should be used in conjunction with other model evaluation metrics, such as accuracy, precision, recall, and F1-score.
#
# Here's some sample Python code to calculate pseudo-R-squared using `scikit-learn`:
#
# We generate 1000 samples with 5 features each using a normal distribution.
#
# We generate the response variable y using a logistic function, which takes the dot product of the features and a set of coefficients, adds some noise, and then applies the logistic function to obtain probabilities. We then threshold these probabilities to obtain binary values (0/1).
#
# We use this generated data as input to the Logistic Regression model.

# %% editable=true slideshow={"slide_type": ""}
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd

# Set the seed for reproducibility
np.random.seed(42)

# Generate X values (features)
n_samples = 1000
n_features = 5

X = np.random.normal(loc=0, scale=1, size=(n_samples, n_features))

# Generate y values (response variable)
# Let's assume the response variable is binary (0/1)
# We'll use a simple logistic function to generate y values
z = np.dot(X, np.array([1, 2, -1, 0.5, -0.5])) + np.random.normal(loc=0, scale=1, size=n_samples)
y = (1 / (1 + np.exp(-z))) > 0.5
y = y.astype(int)

# Create a Pandas DataFrame
df = pd.DataFrame(X, columns=[f'Feature {i+1}' for i in range(n_features)])
df['Target'] = y

print(df.head())

# Assume X is your feature matrix and y is your response variable

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

logreg = LogisticRegression()
logreg.fit(X_train, y_train)

y_pred_proba = logreg.predict_proba(X_test)[:, 1]

auc = roc_auc_score(y_test, y_pred_proba)

# Calculate Cox-Snell R-squared
r2_cs = 1 - (np.log(y_test.shape[0]) - np.sum(y_test * np.log(y_pred_proba) + (1-y_test) * np.log(1-y_pred_proba))) / y_test.shape[0]

print("Cox-Snell R-squared:", r2_cs)

# %% [markdown]
# ### How do you evaluate the performance of a Logistic Regression model?
#
# ### What is the difference between accuracy, precision, recall, and F1-score in Logistic Regression?
#
# ### How do you handle class imbalance in Logistic Regression?
#
# ## Model Building and Assumptions
# 1. What are the assumptions of Logistic Regression?
# 2. How do you check for multicollinearity in Logistic Regression?
# 3. What is the concept of interaction terms in Logistic Regression?
# 4. How do you handle missing values in Logistic Regression?
# 5. What is the difference between a simple and a multiple Logistic Regression model?
#
# ## Applications and Case Studies
# 1. What are some common applications of Logistic Regression?
# 2. How is Logistic Regression used in credit risk assessment?
# 3. What role does Logistic Regression play in medical diagnosis?
# 4. How is Logistic Regression used in marketing and customer churn prediction?
# 5. What are some common challenges faced when implementing Logistic Regression in real-world scenarios?
#
# ## Advanced Topics
# 1. What is the concept of regularization in Logistic Regression?
# 2. How does L1 and L2 regularization differ in Logistic Regression?
# 3. What is the concept of elastic net regularization in Logistic Regression?
# 4. How does Logistic Regression relate to other machine learning algorithms, such as Decision Trees and Random Forests?
# 5. What are some advanced techniques for handling high-dimensional data in Logistic Regression?
