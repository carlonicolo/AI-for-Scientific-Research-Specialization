#!/usr/bin/env python
# coding: utf-8

# In[ ]:


'''
Define a function called 'dictionary_creator' to take in two lists and return a dictionary
with key-value pair list1:list2. If the two lists are not the same size, return the word "Error"
instead. Use a for loop. For those who are interested, there is a way to solve this problem using
'dictionary comprehensions', which are similar in spirit to list comprehensions. 
'''
### dictionary_creator(['a', 'b', 'c'], [1,2,3]) should return {'a': 1, 'b': 2, 'c': 3}
### dictionary_creator(['monkeys', 'pirates', 'captains'], [1,2]) should return "Error"
def dictionary_creator(list1, list2):
    if len(list1) != len(list2):
        return "Error"
    else:
        temp_dict = {}
        for i in range(len(list1)):
            temp_dict[list1[i]] = list2[i]
        return temp_dict

