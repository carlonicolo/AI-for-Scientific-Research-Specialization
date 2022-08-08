#!/usr/bin/env python
# coding: utf-8

# ## K-means clustering
# 
# In this notebook, we'll cluster sequences to find similar sequences with similar patterns.

# In[ ]:


### Loading in libraries and packages

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans


# In[ ]:


### Read in data. We'll store our data in a dataframe called 'df'

df = pd.read_csv('Alignment-HitTable.csv', header = None)
df.columns = ['query acc.verr', 'subject acc.ver', '% identity', 'alignment length', 'mismatches', 
             'gap opens', 'q. start', 'q. end', 's. start', 's. end', 'evalue', 'bit score']
df.head()


# In[ ]:


## Part A (25 pts)

## Fit a K-means clustering with 5 clusters and a random state of 10 on the numeric columns in the dataframe.
## Store the predicted groups in a variable called 'y_pred'. 

# your code here
df_numeric = df[['% identity', 'alignment length', 'mismatches', 
             'gap opens', 'q. start', 'q. end', 's. start', 's. end', 'evalue', 'bit score']]

kmeans = KMeans(n_clusters=5, random_state=10).fit(df_numeric)
y_pred = kmeans.predict(df_numeric)
cluster_labels = kmeans.fit_predict(df_numeric)


# In[ ]:





# In[ ]:


### Part B (15 pts)

## Store the silhouette score on the predicted groups in a variable called 'score'.
## Hint: Use https://scikit-learn.org/stable/modules/generated/sklearn.metrics.silhouette_score.html

# your code here
from sklearn.metrics import silhouette_score
score = silhouette_score(df_numeric,labels=cluster_labels, random_state=10)
score


# In[ ]:





# In[ ]:


## Part C (30 pts)

## Store the silhouette scores for clusters 2 to 9 in a list called 'silhouette_scores'.
## Use a random state of 0 for each prediction.

# your code here
silhouette_scores = []
for i in range(2,10):
    clusterer = KMeans(n_clusters=i, random_state=0)
    cluster_labels= clusterer.fit_predict(df_numeric)
    x = silhouette_score(df_numeric, cluster_labels, random_state = 0)
    silhouette_scores.append(x)


# In[ ]:


plt.bar(range(2, len(silhouette_scores) +2), silhouette_scores)
plt.show()


# In[ ]:


## Part D (30 pts)

## Use a K-means clustering with 5 clusters on the normalized numeric dataframe. Use a random state of 0.
## Hint: Use the https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html to scale the data.
## Store the cluster centers in a dataframe called 'cluster_centers'. 
## Use the index ['Cluster 1', 'Cluster 2', 'Cluster 3', 'Cluster 4', 'Cluster 5'] on the dataframe.

# your code here
from sklearn.preprocessing import StandardScaler

ss = StandardScaler()
df_transformed = ss.fit_transform(df_numeric)
df_transformed = pd.DataFrame(df_transformed)

kmeans = KMeans(n_clusters=5, random_state=0)
kmeans.fit(df_transformed)
cluster_centers = pd.DataFrame(kmeans.cluster_centers_, columns=df.columns[2:], index=['Cluster 1', 'Cluster 2', 'Cluster 3', 'Cluster 4', 'Cluster 5'])
cluster_centers.head()


# In[ ]:




