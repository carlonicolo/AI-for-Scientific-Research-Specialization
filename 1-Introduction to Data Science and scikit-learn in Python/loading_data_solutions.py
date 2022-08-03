#!/usr/bin/env python
# coding: utf-8

# In[1]:


'''
A) Find the breast cancer dataset on the following page: https://scikit-learn.org/stable/modules/classes.html#module-sklearn.datasets
Store the features into the variable 'X' and the target into the varaible 'Y'.
Hint: Make sure to import the dataset before loading it.
'''
from sklearn.datasets import load_breast_cancer
X, y = load_breast_cancer(as_frame = True, return_X_y=True)


# In[ ]:


'''
B) What percentage of people in this dataset have a malignant tumor?
Store the answer into a variable called 'mean_cancer'
'''
mean_cancer = y.mean()


# In[ ]:



'''
C) Transform the 'mean_radius' column to be between 0 and 1 by dividing by it's sum.
We'll perform fancier feature normalization in the next module, but this is a valid approach.
'''
X['mean radius'] = X['mean radius'] / X['mean radius'].sum()

