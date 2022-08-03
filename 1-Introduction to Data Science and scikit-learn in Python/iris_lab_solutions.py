#!/usr/bin/env python
# coding: utf-8

# In[2]:


# importing modules
from sklearn import datasets
import pandas as pd
import numpy as np


# We can load in the iris dataset directly from sklearn.
iris = datasets.load_iris()

# Part 1
'''
a) Create a pandas dataframe called 'data' that combines the petal and sepal data with species information 
The five columns should be 'sepal_length', 'sepal_width', 'petal_length', 'petal_width', and 'species'
'''
data = pd.DataFrame(iris.data)
data.columns = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width'] 
data['species'] = iris.target

'''
b) Notice how the 'species' column is just numbers. The names of the species are stored in a list called target_names.
Replace the species number with the corresponding name from the target_names list.
HINT: We can use the df.replace() function with a map dictionary.
'''
map_dict = {}
for i in range(0,len(iris.target_names)):
    map_dict[i] = iris.target_names[i] 
data['species'] = data.species.replace(map_dict)

# Part 2
'''
a) First, we have to create two new columns, 'sepal_length/width'
and 'petal length/width' to get a sense of the ratios. Create these columns in our 'data' dataframe.
'''
data['sepal_length/width'] = data['sepal_length'] / data['sepal_width']
data['petal_length/width'] = data['petal_length'] / data['petal_width']

'''
b) Now, use the transform function to find the 'mean_sepal_length/width'
and the 'mean_petal_length/width' for each species. You should create these columns
in the 'data' dataframe.
'''
data['mean_sepal_length/width'] = data.groupby('species')['sepal_length/width'].transform('mean')
data['mean_petal_length/width'] = data.groupby('species')['petal_length/width'].transform('mean')

'''
c) Find the squared difference between each row's length/width value and the average of its species.
Create two new columns titled 'sq_diff_sepal_length/width' and ''sq_diff_petal_length/width'. 
HINT: Use the function np.square()
'''
data['sq_diff_sepal_length/width'] = np.square(data['sepal_length/width'] - data['mean_sepal_length/width'])
data['sq_diff_petal_length/width'] = np.square(data['petal_length/width'] - data['mean_petal_length/width'])

# Part 3
'''
a) Create a new dataframe called 'outliers_sepal' and 'outliers_petal' with the top 20 values that deviate most.
HINT: Use the sort_values() function.
'''
outliers_sepal = data.sort_values(by = 'sq_diff_sepal_length/width', ascending = False).head(20)
outliers_petal = data.sort_values(by = 'sq_diff_petal_length/width', ascending = False).head(20)

'''
b) Do any of the rows overlap? Create a list called 'outliers' to store the index numbers of any overlapping rows.
HINT: Use list comprehension.
'''
outliers = [x for x in outliers_petal.index.values if x in outliers_sepal.index.values]

