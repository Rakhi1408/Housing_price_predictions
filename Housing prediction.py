#!/usr/bin/env python
# coding: utf-8

# In[3]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')

from warnings import filterwarnings
filterwarnings(action='ignore')


# In[4]:


housing = pd.read_csv("USA_Housing.csv")


# In[5]:


housing.head()


# In[6]:


housing.info()


# In[7]:


housing.describe()


# In[10]:


housing.columns


# In[11]:


sns.pairplot(housing)


# In[12]:


sns.distplot(housing['Price'])


# In[13]:


sns.heatmap(housing.corr())


# In[14]:


x = housing[['Avg. Area Income', 'Avg. Area House Age', 'Avg. Area Number of Rooms', 
             'Avg. Area Number of Bedrooms', 'Area Population']]
y = housing['Price']


# In[15]:


from sklearn.model_selection import train_test_split


# In[16]:


x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.4, random_state=101)


# In[17]:


from sklearn.linear_model import LinearRegression


# In[18]:


lm = LinearRegression()
lm.fit(x_train,y_train)


# In[19]:


print(lm.intercept_)


# In[20]:


coeff_df = pd.DataFrame(lm.coef_, x.columns,columns=['Coefficient'])
coeff_df


# In[21]:


predictions = lm.predict(x_test)


# In[22]:


plt.scatter(y_test,predictions)


# In[24]:


sns.displot((y_test-predictions),bins=50)


# In[26]:


from sklearn import metrics


# In[28]:


print('MAE:',metrics.mean_absolute_error(y_test, predictions))
print('MSE:',metrics.mean_squared_error(y_test, predictions))
print('RMSE:',np.sqrt(metrics.mean_squared_error(y_test, predictions)))


# In[ ]:




