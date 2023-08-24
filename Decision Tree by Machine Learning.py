#!/usr/bin/env python
# coding: utf-8

# # Decision Tree by Machine Learning

# In[26]:


import pandas as pd
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[27]:


from sklearn.datasets import load_iris


# In[28]:


iris=load_iris()


# In[30]:


iris


# In[31]:


iris.data


# In[32]:


iris.target


# In[33]:


import seaborn as sns


# In[34]:


df=sns.load_dataset('iris')


# In[35]:


df.head()


# In[36]:


#independent features and dependent features
X=df.iloc[:,:-1]
y=iris.target


# In[37]:


X,y


# In[38]:


###train test split
from sklearn.model_selection import train_test_split
X_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)


# In[39]:


X_train


# In[40]:


from sklearn.tree import DecisionTreeClassifier


# In[41]:


##Postpruning
treemodel=DecisionTreeClassifier(max_depth=2)


# In[42]:


treemodel.fit(X_train, y_train)


# In[43]:


from sklearn import tree
plt.figure(figsize=(15,10))
tree.plot_tree(treemodel,filled=True)


# In[44]:


#prediction
y_pred=treemodel.predict(x_test)


# In[45]:


y_pred


# In[46]:


from sklearn.metrics import classification_report, accuracy_score


# In[47]:


score=accuracy_score(y_pred,y_test)


# In[48]:


print(score)


# In[51]:


print(classification_report(y_pred, y_test))

