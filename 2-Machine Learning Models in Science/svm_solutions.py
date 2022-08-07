#!/usr/bin/env python
# coding: utf-8

# In[7]:


### Importing Packages
from sklearn.datasets import load_breast_cancer
data = load_breast_cancer()


# In[8]:


# Part 1
'''
Store the feature data into a variable called 'X' and the target data into a variable called 'y'.
'''
X = data.data
y = data.target


# In[9]:


# Part 2
'''
Fit a linear SVM classifier (from the scikit-learn library) on our dataset. 
After the model is fit, use it to generate predictions on our dataset and saved it in a variable called 'y_pred'.
Hint: Make sure to import the right module first. You'll want to use the classifer found here:
https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html
'''
from sklearn import svm
clf = svm.SVC()
clf.fit(X,y)
y_pred = clf.predict(X)


# In[ ]:




