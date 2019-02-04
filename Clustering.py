#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
os.chdir('C:\\Analytics\\MachineLearning')
#churn = pd.read_csv('Churn_MV.csv')


# In[2]:


churn.head()


# In[3]:


churn = churn.dropna(axis=0, how='all')


# In[4]:


churn.info()


# In[5]:


del churn['Daily Charges MV']


# In[6]:


print(churn)


# In[7]:


x = churn.drop('Churn', axis=1)


# In[8]:


x = x.drop('Phone', axis = 1)


# In[9]:


x = x.drop(['State', 'Area Code'], axis = 1)


# In[10]:


for i in x:
    x[i] = (x[i] - x[i].min())/(x[i].max() - x[i].min())


# In[11]:


x


# In[12]:


from sklearn.cluster import KMeans


# In[13]:


churn['Frequency'] = churn.groupby(['State','Area Code']).transform('count')['Account Length']


# In[14]:


churn['Frequency']


# In[15]:


from sklearn.cluster import KMeans


# In[20]:


freq = churn['Frequency']


# In[23]:


freq = np.array(freq)

freq = freq.reshape(-1,1)


# In[24]:


wcss = []

for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, init = 'k-means++', max_iter = 300, n_init = 10, random_state=0)
    kmeans.fit(freq)
    wcss.append(kmeans.inertia_)


# In[25]:


get_ipython().run_line_magic('matplotlib', 'inline')


# In[26]:


plt.plot(range(1,11), wcss)
plt.show()


# In[27]:


kmeans = KMeans(n_clusters = 4, init = 'k-means++', max_iter=300, n_init=10, random_state=0)
clusters = kmeans.fit_predict(x)


# In[28]:


clusters = pd.DataFrame(clusters)
clusters


# In[29]:


churn.index = (churn.index-1)/2


# In[30]:


churn['Freq Cluster'] = clusters


# In[31]:


churn


# In[34]:


churn_freq = churn.groupby(['State', 'Area Code','Churn']).count().reset_index()

churn_freq['Target_freq'] = churn_freq['Account Length']


# In[98]:


dummy_freq = churn.groupby(['State', 'Area Code']).count()['Account Length'].reset_index(level=['State','Area Code'])
dummy_freq['dummyFreq'] = dummy_freq['Account Length']
dummy_freq = dummy_freq.drop(['Account Length'], axis=1)


# In[99]:


pd.merge(churn, dummy_freq, how='inner', on=['State', 'Area Code'])


# In[111]:


churn['State'].value_counts()


# In[ ]:





# In[73]:


f = churn.groupby(['State', 'Area Code', 'Churn']).count()['Account Length']
f = f.unstack('Churn')
f = f.fillna(0)
f = f.reset_index(level=['State','Area Code'])
churn = pd.merge(churn, f, how='inner', on=['State','Area Code'])


# In[74]:


churn


# In[75]:


churn['Freq_churn0'] = churn[0.0]


# In[76]:


churn['Freq_churn1'] = churn[1.0]


# In[79]:


churn = churn.drop([0.0,1.0], axis=1)


# In[81]:


x = churn[['Freq_churn0', 'Freq_churn1']]
x


# In[82]:


wcss = []

for i in range(1, 11):
    kmeans = KMeans(n_clusters = i, init = 'k-means++', max_iter = 10, random_state = 0)
    kmeans.fit(x)
    wcss.append(kmeans.inertia_)


# In[83]:


plt.plot(range(1, 11), wcss)


# So from the above chart we can take number of clusters as 4 or 5

# In[ ]:




