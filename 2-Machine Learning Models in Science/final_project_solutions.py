#!/usr/bin/env python
# coding: utf-8

# In[1]:


### Importing modules and loading in data

import sklearn
import numpy 
from sklearn import datasets

data = datasets.load_diabetes()


# In[2]:


## Part A (20 pts)

'''
Use a train/test split to divide the data into X_train, X_test, y_train, and y_test 
correpsonding to the input features and the output vector. 
For testing purposes, use a random state of 10.
'''

from sklearn.model_selection import train_test_split

X = data['data']
y = data['target']

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=10)


# In[3]:


## Part B (20 pts)

'''
Run a simple linear regresison algorithm on the input data. 
Store the fitted model in the variable 'reg'.

Hint: You mauy refer to the documentation on the following page: 
https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html
'''

from sklearn.linear_model import LinearRegression
reg = LinearRegression().fit(X_train, y_train)


# In[4]:


## Part C (20 pts)

'''
Run a random forest regressor on the input data. 
Store the fitted model in the variable 'rf'.
Use a max_depth of 5 and a random state of 42.

Hint: You may refer to the documentation on the following page: 
https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html
'''

from sklearn.ensemble import RandomForestRegressor
rf = RandomForestRegressor(max_depth=5, random_state=42).fit(X_train, y_train)


# In[5]:


## Part D (20 pts)

'''
Run the MLP regresison algorithm on the input data. 
Use a hidden layer size of 10, stochastic gradient descent solver, and a random state of 42.
Store the fitted model in the variable 'nn'. 

Hint: You may refer to the docomentation on the following page: 
https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPRegressor.html#sklearn.neural_network.MLPRegressor
'''

from sklearn.neural_network import MLPRegressor
nn = MLPRegressor(hidden_layer_sizes = 10, solver = 'sgd', random_state=42).fit(X_train, y_train)


# In[7]:


## Part E (20 pts)

'''
Let's compare the results of the three algorithms on the test set.
There are numerous ways we can compare accuracy, but for the purposes of this
exercise we're going to use the built in 'score' method from the scikit-learn library.

Create a new variable called 'best'. 
Set it equal to 0 if the linear regressor performes best,
1 if the random forest performs best, or
2 if the neural network performs best.

Hint: You may refer the 'score' method at the following page:
https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html#sklearn.linear_model.LinearRegression.score
'''

acc_nn = nn.score(X_test, y_test)
acc_reg = reg.score(X_test, y_test)
acc_rf = rf.score(X_test, y_test)
best = 0


# In[8]:


acc_nn


# In[9]:


acc_reg


# In[10]:


acc_rf


# In[ ]:




