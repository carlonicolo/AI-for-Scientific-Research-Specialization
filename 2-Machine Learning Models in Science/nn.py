#!/usr/bin/env python
# coding: utf-8

# ## Neural Networks
# 
# In this practice assignment, we'll swap out linear classifiers for neural networks to predict the presence of breast cancer amongst patients.

# In[ ]:


### Loading packages and importing data

from sklearn import datasets
import numpy as np
import sklearn

data = datasets.load_breast_cancer()


# In[ ]:


## Part A (25 pts)

'''
When creating models, it's important to split our data into a train and test set.
You can read more about it here: https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html
Store the data into X_train, X_test, y_train, and y_test arrays correpsonding to the input features and the output vector.
Use a random state of 1. 
We have already imported the module for you
'''
from sklearn.model_selection import train_test_split

# your code here


# In[ ]:



''' 
End of Section
'''


# In[ ]:


## Part B (50 pts)

'''
Run a MLP classifier on the input data. 
Use a hidden layer size of 30, stochastic gradient descent solver, and a random state of 42.
Store the fitted model in the variable 'nn'. 

Hint: Look at the documentation on the following page: 
https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPClassifier.html#sklearn.neural_network.MLPRegressor
'''

# your code here


# In[ ]:



''' 
End of Section
'''


# In[ ]:


## Part C (25 pts)

'''
Make predictions on the test set using the fitted network.
Store the predictions in a variable called 'pred'.
Find the accuracy of the model on the test set and store in a variable called 'accuracy'.
'''

# your code here


# In[ ]:



''' 
End of Section
'''

