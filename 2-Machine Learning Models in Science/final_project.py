#!/usr/bin/env python
# coding: utf-8

# ### Final Project
# 
# In this project, we're going to compare the performance of various models in predicting the presence of diabetes amongs patients.

# In[ ]:


### Importing modules and loading in data

import sklearn
import numpy 
from sklearn import datasets

data = datasets.load_diabetes()


# In[ ]:


## Part A (20 pts)

'''
Use a train/test split to divide the data into X_train, X_test, y_train, and y_test 
correpsonding to the input features and the output vector. 
For testing purposes, use a random state of 10.
'''

from sklearn.model_selection import train_test_split

# your code here


# In[ ]:



''' 
End of Section
'''


# In[ ]:


## Part B (20 pts)

'''
Run a simple linear regresison algorithm on the input data. 
Store the fitted model in the variable 'reg'.

Hint: You mauy refer to the documentation on the following page: 
https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html
'''

# your code here


# In[ ]:



''' 
End of Section
'''


# In[ ]:


## Part C (20 pts)

'''
Run a random forest regressor on the input data. 
Store the fitted model in the variable 'rf'.
Use a max_depth of 5 and a random state of 42.

Hint: You may refer to the documentation on the following page: 
https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html
'''

# your code here


# In[ ]:


### BEGIN HIDDEN_TESTS
assert rf.predict(X)[10] == 101.50989829460126
### END HIDDEN_TESTS

''' 
End of Section
'''


# In[ ]:


## Part D (20 pts)

'''
Run the MLP regresison algorithm on the input data. 
Use a hidden layer size of 10, stochastic gradient descent solver, and a random state of 42.
Store the fitted model in the variable 'nn'. 

Hint: You may refer to the docomentation on the following page: 
https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPRegressor.html#sklearn.neural_network.MLPRegressor
'''

# your code here


# In[ ]:



''' 
End of Section
'''


# In[ ]:


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

# your code here


# In[ ]:



''' 
End of Section
'''

