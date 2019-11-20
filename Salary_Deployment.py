# -*- coding: utf-8 -*-
"""
Created on Wed Nov 20 11:29:10 2019

@author: Sabitha Jaleel
"""

# Simple Linear Regression

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Salary_Data.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 1].values



# Fitting Simple Linear Regression to the Training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X, y)

import pickle
 pickle.dump(regressor,open('Salary_Deployment.pkl','wb'))

