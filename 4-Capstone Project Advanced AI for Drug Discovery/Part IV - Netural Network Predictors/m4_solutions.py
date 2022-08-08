#!/usr/bin/env python
# coding: utf-8

# In[ ]:


### Loading in data

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


# In[ ]:


### Reading in data. We'll store our data in dataframe called 'df'.

df = pd.read_csv('Alignment-HitTable.csv', header = None)
df.columns = ['query acc.verr', 'subject acc.ver', '% identity', 'alignment length', 'mismatches', 
             'gap opens', 'q. start', 'q. end', 's. start', 's. end', 'evalue', 'bit score']
df.head()


# In[ ]:


## Part A (25 pts)

## In this final notebook, we'll be predicting 'bit_score' from some of the columns in the data.
## Create a feature dataframe called 'X' with the columns: ['% identity',  'mismatches', 'gap opens', 'q. start', 's. start'].
## Store the target 'y' with the bit scores.

X = df[['% identity',  'mismatches', 
             'gap opens', 'q. start', 's. start']]
y = df['bit score']


# In[ ]:


## Part B (15 pts)

### Use the Standard Scaler from the sklearn.preprocessing library to normalize the data.
## Store the transformed X data in a variable called 'X_transformed'.

from sklearn.preprocessing import StandardScaler

ss = StandardScaler()
X_transformed = ss.fit_transform(X)


# In[ ]:


## Part C (25 pts)

## Predict the bit score by fitting a linear regression model. Store the predicted bit scores in
## a variable called 'lin_pred'. Get the score in a variable called 'lin_score'. 
## Store the linear regression coefficients in a variable called 'coef'.

from sklearn.linear_model import LinearRegression

linreg = LinearRegression()
linreg.fit(X_transformed, y)
score = linreg.score(X_transformed, y)
lin_pred = linreg.predict(X_transformed)
coef = linreg.coef_


# In[ ]:


## Part D (35 pts)

## Split the data into a train/test set using the sklearn train/test split with a random state of 42.
## Predict the bit score by fitting a neural network MP regression model. 
## Use a random state of 42, 3 hidden layers each of size 50, and a 'sgd' solver.
## Store the predicted bit scores of the test set in a variable called 'nn_pred'. 
## Get the score on the test set in a variable called 'nn_score'. 

from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor

X_train, X_test, y_train, y_test = train_test_split(X_transformed, y, random_state = 42)
nn = MLPRegressor(random_state=42, solver = 'sgd', hidden_layer_sizes = (50,3)).fit(X_train, y_train)
nn_pred = nn.predict(X_test)
nn_score = nn.score(X_test, y_test)

