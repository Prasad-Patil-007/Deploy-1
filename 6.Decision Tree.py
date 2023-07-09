#!/usr/bin/env python
# coding: utf-8

# In[41]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split

import warnings
warnings.filterwarnings("ignore")


# In[35]:


df_iris = pd.read_csv("IRIS.csv")
df_iris.head()


# In[36]:


df_iris.shape


# In[37]:


df_iris.info()


# In[38]:


df_iris.isnull().sum()


# In[42]:


sns.countplot(df_iris['species'])


# In[22]:


sns.pairplot(df_iris)


# In[43]:


df_iris.corr()


# In[44]:


X = df_iris.drop('species',axis=1)
X.shape


# In[46]:


y = df_iris['species']
y.shape


# In[47]:


y.value_counts()


# In[48]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=22)


# In[49]:


dt_model = DecisionTreeClassifier(max_depth=2,random_state=42,criterion='entropy',min_samples_split=2)
dt_model.fit(X_train,y_train)


# In[50]:


from sklearn.tree import plot_tree
plt.figure(figsize=(5,5), dpi=150)
plot_tree(dt_model, feature_names=X.columns,class_names=y);


# In[ ]:





# In[ ]:




