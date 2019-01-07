# -*- coding: utf-8 -*-

# Importing Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Importing the Dataset
dataset = pd.read_csv('mediaCompanyData.csv')
dataset = dataset.drop('Unnamed: 7', axis = 1)
dataset['Date'] = pd.to_datetime(dataset['Date']).dt.date
y = dataset.iloc[:, 1].values

# Deriving "days since the show started"
from datetime import date
d0 = date(2017, 2, 28)
d1 = dataset.Date
delta = d1 - d0
delta = delta.astype(str)
delta = delta.map(lambda x: x[0:2])
delta = delta.astype(int)
dataset['day']= delta

# Independent Variable
X = dataset.drop('Views_show', axis = 1).values
X = X[:, 1:].astype(float)

"""# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X = sc_X.fit_transform(X)
sc_y = StandardScaler()
y = sc_y.fit_transform(y)"""

# Fitting Multiple Linear Regression model to dataset
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X, y)

# Building a optimal model using Backward Elimination
import statsmodels.formula.api as sm
X_opt = X[:, [0, 1, 2, 3, 4, 5]]
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
regressor_OLS.summary()

X_opt = X[:, [0, 1, 2, 4, 5]]
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
regressor_OLS.summary()

X_opt = X[:, [0, 1, 2, 5]]
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
regressor_OLS.summary()

X_opt = X[:, [0, 1, 2, 5]]
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
regressor_OLS.summary()

X_opt = X[:, [1, 2, 5]]
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
regressor_OLS.summary()

X_opt = X[:, [2, 5]]
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
regressor_OLS.summary()
