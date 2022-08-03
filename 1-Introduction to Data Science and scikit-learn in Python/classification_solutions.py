#!/usr/bin/env python
# coding: utf-8

# In[1]:


### Importing Files
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.linear_model import LogisticRegression

data = pd.read_csv("heart.csv", index_col = 0)
data.head()

# Part 1
'''
a) Use one-hot encoding to transform the 'thal' feature into two columns called 'is_normal', 'is_fixed', 
and 'is_reversible'. (15 pts). Be sure to drop the 'thal' column afterwards.
Hint: https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.get_dummies.html
'''
data['thal'].replace({3: 'normal', 6: 'fixed', 7: 'reversible'}, inplace = True)
data = pd.get_dummies(data, columns=["thal"], prefix=["is"])

'''
b) Use min-max normalzaition to resacle all the features between 0 and 1 (15 pts). Make sure that data remains in the same
dataframe format.
Hint: Use https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.MinMaxScaler.html
'''
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
data = pd.DataFrame(scaler.fit_transform(data.values), columns=data.columns, index=data.index)

'''
c) Split the data into a train, test set using a 75/25 split. Use a random state of 42 for grading purposes (20 pts).
Hint: Use https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html
'''
X = data.drop(columns = 'target')
y = data['target']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# Part 2
'''
a) Fit a logisitic regression classifier on the data. Save the model in a varaible called 'clf'. Use a random state of 42.
Use the following paramters: penalty:'l2', solver:'liblinear', C:0.1. 15 pts.
'''
clf = LogisticRegression(penalty='l2', solver='liblinear', C=0.1, random_state = 42)
clf.fit(X_train, y_train)

'''
b) Generate 0/1 predictions on the test set and store them in a varaible called 'pred'. 
Generate proabbility prerdictions on the test set and store them in a variable called 'scores'.
10 pts
'''
pred = clf.predict(X_test)
scores = clf.predict_proba(X_test)[:,1]

'''
c) Fill in this function to find and return the root mean sqaured error between the predicted and actual values.
Hint: Use his formula for the rsme: https://sciencing.com/calculate-mean-deviation-7152540.html.
10 pts
'''
def rsme(predictions, actuals):
    from sklearn.metrics import mean_squared_error
    return mean_squared_error(actuals, predictions, squared=False)

'''
d) Try using a random forest classifier to fit the data instead. Use the default paramters and a random state of 42.
Save the fitted model into a varaible called 'rf'. Generate the 'pred' and 'scores' in a similar way to part b.
Hint: Use https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html
15 pts
'''
from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(random_state=42)
rf.fit(X_train, y_train)
pred = rf.predict(X_test)
scores = rf.predict_proba(X_test)[:,1]


# In[2]:


pred


# In[3]:


scores


# In[ ]:




