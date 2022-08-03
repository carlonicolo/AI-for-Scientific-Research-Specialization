#!/usr/bin/env python
# coding: utf-8

# ## Dictionaries
# 
# a) 1 pt
# 
# b) 1 pt
# 
# c) 1 pt

# In[1]:


'''
a) Create a dictionary called "rotation_speed" with the key-value pair planet:speed.

Planet
Mercury:    6.73
Venus:      4.05
Earth:      1,040
Mars:       538
Jupiter:    28,325
Saturn:     22,892
Uranus:     9,193
Neptune:    6,039
'''

# your code here
rotation_speed = {'Mercury': 6.73, 'Venus': 4.05, 'Earth':  1040, 'Mars':  538, 'Jupiter': 28325,
'Saturn':  22892, 'Uranus': 9193, 'Neptune':  6039}


# In[2]:


'''
END OF SECTION
'''


# In[3]:


'''
b) Store the keys of the dictionary into a list called 'planets'. Store the values into a list called 'speeds'.
HINT: Be sure that these variables are stored as lists. You can use list(x) on any lisst-like object x to convert it into a list.
'''
# your code here
planets = list(rotation_speed.keys())
speeds = list(rotation_speed.values())


# In[4]:


'''
END OF SECTION
'''


# In[5]:


'''
c) Extract the supply for 'Mars' and store it into a variable called 'mars_speed'
'''
# your code here
mars_speed = rotation_speed['Mars']


# In[6]:


'''
END OF SECTION
'''


# In[ ]:




