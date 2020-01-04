#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


df = pd.read_csv('E:/Python/training_internshala/data_cleaned.csv')


# In[3]:


df.shape


# In[4]:


df.head()


# In[5]:


#Seperating independent and dependent variables

x = df.drop(['Survived'], axis = 1)
y = df['Survived']


# In[6]:


from sklearn.model_selection import train_test_split


# In[7]:


train_x, test_x, train_y, test_y  = train_test_split(x,y, random_state = 101, stratify = y)


# In[8]:


train_y.value_counts()/len(train_y)


# In[9]:


test_y.value_counts()/len(test_y)


# In[10]:


# import decision tree classifier

from sklearn.tree import DecisionTreeClassifier


# In[11]:


clf = DecisionTreeClassifier()


# In[12]:


clf.fit(train_x, train_y)


# In[13]:


clf.score(train_x, train_y)


# In[14]:


clf.score(test_x, test_y)


# In[15]:


clf.predict(test_x)


# In[ ]:





# In[ ]:




