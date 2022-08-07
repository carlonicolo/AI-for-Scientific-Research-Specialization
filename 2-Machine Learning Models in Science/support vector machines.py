#!/usr/bin/env python
# coding: utf-8

# ### Support Vector Machines
# 
# In this notebook, we look at support vector machines and their use as linear models.

# In[13]:


### Let's start by importing standard libraries.

get_ipython().run_line_magic('matplotlib', 'inline')
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import seaborn as sns
from sklearn.svm import SVC


# Read the following article to build some intuition behind support vectors and thier use in machine learning.
# 
# Link to article: https://towardsdatascience.com/support-vector-machine-introduction-to-machine-learning-algorithms-934a444fca47

# In[ ]:


### hard vs. soft svm
### logistic regression, cutoff - compare results
### Why svm instead? - kernel trick
### math of kernel trick works really well with SVM
### high-deminesional embedding and create hyperplane

### https://stats.stackexchange.com/questions/95340/comparing-svm-and-logistic-regression


# In[3]:


### Let's create some classes to use as our dataset.
### We can do so with the 'make_blobs' function

from sklearn.datasets import make_blobs
X, y = make_blobs(n_samples=50, centers=2,
                  random_state=0, cluster_std=2.0)
plt.scatter(X[:, 0], X[:, 1], c=y, s=50, cmap='autumn');


# In[6]:


### Let's fit a line to separate the boundary of the classes.
### Note that we have some options on which line we could use to separate the classes.

xfit = np.linspace(-1, 3.5)
plt.scatter(X[:, 0], X[:, 1], c=y, s=50, cmap='autumn')
plt.plot([0.6], [2.1], 'x', color='red', markeredgewidth=2, markersize=10)

for m, b in [(1, 0.65), (0.5, 1.6), (-0.2, 2.9)]:
    plt.plot(xfit, m * xfit + b, '-k')

plt.xlim(-1, 3.5);


# In[9]:


### This function is just used for plotting. Feel free to skip past it.


def plot_svc_decision_function(model, ax=None, plot_support=True):
    """Plot the decision function for a 2D SVC"""
    if ax is None:
        ax = plt.gca()
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    
    # create grid to evaluate model
    x = np.linspace(xlim[0], xlim[1], 30)
    y = np.linspace(ylim[0], ylim[1], 30)
    Y, X = np.meshgrid(y, x)
    xy = np.vstack([X.ravel(), Y.ravel()]).T
    P = model.decision_function(xy).reshape(X.shape)
    
    # plot decision boundary and margins
    ax.contour(X, Y, P, colors='k',
               levels=[-1, 0, 1], alpha=0.5,
               linestyles=['--', '-', '--'])
    
    # plot support vectors
    if plot_support:
        ax.scatter(model.support_vectors_[:, 0],
                   model.support_vectors_[:, 1],
                   s=300, linewidth=1, facecolors='none');
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)


# In[18]:


### The real beenfit of support vectors come when we alter the kernel. 
### First, let's see the problem with a linear kernel.

from sklearn.datasets import make_circles

X, y = make_circles(100, factor=.1, noise=.1)

clf = SVC(kernel='linear').fit(X, y)
plt.scatter(X[:, 0], X[:, 1], c=y, s=50, cmap='autumn')
plot_svc_decision_function(clf, plot_support=False);


# In[19]:


### Notice that by mapping to a higher-dimensional space, we can create non-linear decision
### boundaries in 2D-space.

clf = SVC(kernel='rbf', C=1E6)
clf.fit(X, y)

plt.scatter(X[:, 0], X[:, 1], c=y, s=50, cmap='autumn')
plot_svc_decision_function(clf)
plt.scatter(clf.support_vectors_[:, 0], clf.support_vectors_[:, 1],
            s=300, lw=1, facecolors='none');


# Optional: For a more detailed picture of support vector machines, you can look at the following article: https://towardsdatascience.com/support-vector-machines-svm-c9ef22815589
