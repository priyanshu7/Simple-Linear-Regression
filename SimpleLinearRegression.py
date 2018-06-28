# -*- coding: utf-8 -*-
"""
Created on Wed Jun 20 22:21:44 2018

@author: priyanshumehta
"""

# Data Preprocessing Template

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Salary_data.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 1].values

# Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 1/3, random_state = 0)

# Feature Scaling
"""from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
sc_y = StandardScaler()
y_train = sc_y.fit_transform(y_train)"""

#Fitting Simple Linear Regression to the Training Set

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

#Predicting the Test Set results
y_pred = regressor.predict(X_test)

#Visualising Training Set Results
plt.scatter(X_train, y_train, color='red')
plt.plot(X_train , regressor.predict(X_train) , color = 'yellow' )
plt.title('SALARY VS EXPERIENCE ( Training Set )')
plt.xlabel('Years Of Experience')
plt.ylabel('Salary')
plt.show()

#Visualising Test Set Results
plt.scatter(X_test, y_test, color='red')
plt.plot(X_train , regressor.predict(X_train) , color = 'yellow' )
plt.title('SALARY VS EXPERIENCE ( Test Set )')
plt.xlabel('Years Of Experience')
plt.ylabel('Salary')
plt.show()

