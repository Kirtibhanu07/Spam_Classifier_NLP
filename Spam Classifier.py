#!/usr/bin/env python
# coding: utf-8

# ## Spam Message Detection

# Data Preprocessing

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


import warnings
warnings.filterwarnings('ignore')


# In[3]:


data=pd.read_csv('Desktop/spam.tsv',sep='\t')


# In[4]:


data.head()


# In[5]:


data.isnull().sum()


# In[6]:


data.describe()


# In[7]:


data['label'].value_counts(normalize=True)


# In[8]:


ham=data[data['label']=='ham']


# In[9]:


spam=data[data['label']=='spam']


# In[10]:


ham.shape


# In[11]:


spam.shape


# In[12]:


ham=ham.sample(spam.shape[0])


# In[13]:


ham.shape ,spam.shape


# In[14]:


df=ham.append(spam,ignore_index=True)


# In[15]:


df.shape


# In[16]:


df['label'].value_counts()


# In[17]:


df.head()


# In[18]:


plt.hist(df[df['label']=='ham']['length'],bins=100,alpha=0.7)
plt.hist(df[df['label']=='spam']['length'],bins=100,alpha=0.7)
plt.show()


# In[19]:


plt.hist(df[df['label']=='ham']['punct'],bins=100,alpha=0.7)
plt.hist(df[df['label']=='spam']['punct'],bins=100,alpha=0.7)
plt.show()


# In[20]:


df


# In[21]:


from sklearn.model_selection import train_test_split


# In[22]:


X_train,X_test,y_train,y_test=train_test_split(df['message'],df['label'],test_size=0.3,random_state=0,shuffle=True)


# In[23]:


from sklearn.feature_extraction.text import TfidfVectorizer


# In[24]:


from sklearn.ensemble import RandomForestClassifier


# In[25]:


from sklearn.pipeline import Pipeline


# In[26]:


classifier=Pipeline([('tfidf',TfidfVectorizer()),('classifier',RandomForestClassifier(n_estimators=10))])


# In[27]:


classifier.fit(X_train,y_train)


# ## predicting the result(RANDOM FOREST)

# In[28]:


y_pred=classifier.predict(X_test)


# In[29]:


y_test,y_pred


# In[30]:


from sklearn.metrics import classification_report,accuracy_score,confusion_matrix


# In[31]:


accuracy_score(y_test,y_pred)


# In[32]:


confusion_matrix(y_test,y_pred)


# In[33]:


print(classification_report(y_test,y_pred))


# ## SVM

# In[34]:


from sklearn.svm import SVC


# In[35]:


svm=Pipeline([('tfidf',TfidfVectorizer()),('classifier',SVC(C=100,gamma='auto'))])


# In[36]:


svm.fit(X_train,y_train)


# In[37]:


y_pred=svm.predict(X_test)


# In[38]:


accuracy_score(y_test,y_pred)


# In[39]:


confusion_matrix(y_test,y_pred)


# In[40]:


print(classification_report(y_test,y_pred))


# In[56]:


test1 = ['Hello, You are learning natural Language Processing']
test3 = ['Hope you are doing good and learning new things !']
test2 = ['Congratulations, You won a lottery ticket worth $1 Million ! To claim call on 446677']


# In[57]:


classifier.predict(test1)


# In[58]:


classifier.predict(test2)


# In[59]:


classifier.predict(test3)


# In[45]:


svm.predict(test1)


# In[46]:


svm.predict(test2)


# In[47]:


svm.predict(test3)


# In[ ]:




