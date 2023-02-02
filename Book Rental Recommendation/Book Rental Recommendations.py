#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns


# In[2]:


ratings = pd.read_csv('BX-Book-Ratings.csv',encoding='latin-1')
books = pd.read_csv('BX-Books.csv',encoding='latin-1')
users = pd.read_csv('BX-Users.csv',encoding='latin-1')
recommend = pd.read_csv('Recommend.csv',encoding='latin-1')


# In[3]:


ratings.head()


# In[4]:


ratings.info()


# In[5]:


ratings.shape


# In[6]:


ratings.isnull().sum()


# In[7]:


ratings = ratings.drop_duplicates()


# In[8]:


ratings.shape


# In[9]:


ratings.size


# In[10]:


ratings['isbn'] = ratings['isbn'].str.extract('(\d+)')


# In[11]:


ratings.info()


# In[12]:


ratings['isbn'] = pd.to_numeric(ratings['isbn'])


# In[13]:


ratings.shape


# In[14]:


ratings = ratings.dropna()


# In[15]:


ratings.shape


# In[16]:


Number_of_users = pd.unique(ratings['user_id'])


# In[17]:


Number_of_users.size


# ## No. of Unique Users is 95502

# In[18]:


books.head()


# In[19]:


books.info()


# In[20]:


books = books.dropna()


# In[21]:


books.info()


# In[22]:


books = books.drop_duplicates()


# In[23]:


books.shape


# In[24]:


books['isbn'] = books['isbn'].str.extract('(\d+)')


# In[25]:


books['isbn'] = pd.to_numeric(books['isbn'])


# In[26]:


books.info()


# In[27]:


Number_of_books = pd.unique(books['book_title'])                          


# In[28]:


Number_of_books.size


# ## Number of books is 242148

# In[29]:


users.head()


# In[30]:


users.info()


# In[31]:


users['Age'] = users['Age'].fillna(users['Age'].mean())


# In[32]:


users.head()


# In[33]:


users['Age'] = users['Age'].astype(int)


# In[34]:


users.head(10)


# In[35]:


users.info()


# In[36]:


users.shape


# In[37]:


users = users.drop_duplicates()


# In[38]:


users.shape


# In[39]:


users.isnull().sum()


# In[40]:


users = users.dropna()


# In[41]:


users['user_id'] = pd.to_numeric(users['user_id'])


# In[42]:


users.info()


# In[43]:


final = pd.merge(ratings,books,on='isbn')


# In[44]:


final.shape


# In[45]:


final_1 = pd.merge(final,users,on='user_id')


# In[46]:


final_1.head()


# In[47]:


final_1[['user_id','isbn']] = final_1[['user_id','isbn']].sort_values(by='user_id')


# In[48]:


final_1.head()


# In[49]:


final_1.info()


# In[50]:


recommend.head()


# In[51]:


columns = ['user_id','user_id2','category','isbn']
recommend.columns = columns


# In[52]:


recommend.head()


# In[53]:


x = recommend[['user_id','user_id2','isbn']]
y = recommend['category']


# In[54]:


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.30)


# In[55]:


from sklearn.linear_model import LinearRegression
lm = LinearRegression()


# In[56]:


model = lm.fit(x_train,y_train)


# In[57]:


y_pred = lm.predict(x_test)


# In[58]:


from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score


# In[59]:


MAE = mean_absolute_error(y_test,y_pred)
MAE


# In[60]:


MSE = mean_squared_error(y_test,y_pred)
MSE


# In[61]:


r2 = r2_score(y_test,y_pred)
r2


# In[62]:


RMSE = np.sqrt(MSE)
RMSE


# In[ ]:




