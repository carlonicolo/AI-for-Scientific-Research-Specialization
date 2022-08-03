#!/usr/bin/env python
# coding: utf-8

# ## Loops
# 
# 
# We have created a diciotnary called plants with the values represening annual supply out of that region.
# Use a for loop to populate a list called 'updated_plants' with only the plant names of the plants with an annual supply of more than 875 units.

# In[2]:


plants = {'Dallas': 900, 'Las Vegas': 500, 'Detroit': 832, 'Nashville': 600, 'Los Angeles': 2400, 'Seattle': 1300, 'Denver': 1800}


# In[15]:


updated_plants = []

for plant in plants:
    
    '''
    Populate the list called 'updated_plants' with only the plant names of the 
    plants with an annual supply of more than 875 units.
    '''
    
    # your code here
    if (plants[plant] > 875 ):
        #print(plant)
        updated_plants.append(plant)


# In[7]:


'''
END OF SECTION
'''


# In[ ]:




