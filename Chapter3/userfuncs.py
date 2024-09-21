from matplotlib.pyplot import subplots
import numpy as np
import pandas as pd
from patsy import dmatrices
from statsmodels.stats.outliers_influence import variance_inflation_factor as VIF
from pandas.api.types import is_numeric_dtype
from scipy import stats
from statsmodels.stats.anova import anova_lm
import statsmodels.formula.api as smf


# Display residuals plot function
def display_residuals_plot(results):
    _, ax = subplots(figsize=(8, 8))
    ax.scatter(results.fittedvalues, results.resid)
    ax.set_xlabel("Fitted values for " + results.model.endog_names)
    ax.set_ylabel("Residuals")
    ax.axhline(0, c="k", ls="--")


# Identify least statistically significant variable or column
def identify_least_significant_feature(results, alpha=0.05):
    index = np.argmax(results.pvalues)
    highest_pvalue = results.pvalues.iloc[index]
    if highest_pvalue > alpha:
        variable = results.pvalues.index[index]
        coeff = results.params.iloc[index]
        print("We find the least significant variable in this model is " +
              variable + " with a p-value of " + str(highest_pvalue) +
              " and a coefficient of " + str(coeff))
        print("Using the backward methodology, we drop " + variable +
              " from the new model")
    else:
        print("No variables are statistically insignificant.")
        print("The model " + results.model.formula +
              " cannot be pruned further.")


# Calculate [Variance Inflation Factors(VIFs) for features in a model](https://www.statology.org/how-to-calculate-vif-in-python/)
def calculate_VIFs(formula, df):
    # find design matrix for linear regression model using formula and dataframe
    _, X = dmatrices(formula, data=df, return_type='dataframe')
    # calculate VIF for each explanatory variable
    vif = pd.DataFrame()
    vif['VIF'] = [VIF(X.values, i) for i in range(1, X.shape[1])]
    vif['Feature'] = X.columns[1:]
    vif = vif.set_index(["Feature"])
    return vif


# Identify feature with highest VIF
def identify_highest_VIF_feature(vifdf, threshold=5):
    highest_vif = vifdf["VIF"].iloc[np.argmax(vifdf)]
    if highest_vif > threshold:
        variable = vifdf.index[np.argmax(vifdf["VIF"])]
        print("We find the highest VIF in this model is " + variable +
              " with a VIF of " + str(highest_vif))
        print("Hence, we drop " + variable + " from the model to be fitted.")
    else:
        print("No variables are significantly collinear.")


# Function to standardize numeric columns
def standardize(series):
    if is_numeric_dtype(series):
        return stats.zscore(series)
    return series


# Function to produce linear regression analysis
def perform_analysis(response, formula, df):
    model = smf.ols(f'{response} ~ {formula}', data=df)
    results = model.fit()
    print(results.summary())
    print(anova_lm(results))
    return results
