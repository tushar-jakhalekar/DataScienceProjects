#!/usr/bin/env python
# coding: utf-8

# ## Importing the important libraries

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


train = pd.read_csv('train.csv')


# In[3]:


train.head()


# In[4]:


train.describe()   ## Getting the insights of data bu .describe function


# In[5]:


train.columns


# In[6]:


train.shape


# ## Variables having zero varience

# In[7]:


train.var()[train.var()==0] 


# ## Remove the variables which have zero varience

# In[8]:


train_new = train.drop(train.var()[train.var()==0].index.values,axis=1)


# In[9]:


train_new.shape


# In[10]:


train_new.isnull().sum()  ## Checking for null values


# In[11]:


train_new = train_new.dropna()   ## Drop the null values if any


# In[12]:


train_new.shape


# In[13]:


train_new.drop_duplicates()


# In[14]:


train_new.columns


# In[15]:


print(len(train_new['y'].unique()))


# In[16]:


train_new.info()   ## .info gives information about the data types in the data


# ## Using the label encoder to encode the string values into numeric categorical format

# In[17]:


from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()


# In[18]:


train_new.select_dtypes(object)


# In[19]:


train_new['X0'] = le.fit_transform(train_new['X0'])
train_new['X1'] = le.fit_transform(train_new['X1'])
train_new['X2'] = le.fit_transform(train_new['X2'])
train_new['X3'] = le.fit_transform(train_new['X3'])
train_new['X4'] = le.fit_transform(train_new['X4'])
train_new['X5'] = le.fit_transform(train_new['X5'])
train_new['X6'] = le.fit_transform(train_new['X6'])
train_new['X8'] = le.fit_transform(train_new['X8'])

# columns = ['X0','X1','X2','X3','X4','X5','X6','X8']
# train_new[columns] = train_new[columns].apply(le.fit_transform)


# In[20]:


train_new.head()


# ## Scaling the data

# In[21]:


from sklearn.preprocessing import StandardScaler
sc = StandardScaler()


# In[22]:


train_scaled = sc.fit_transform(train_new.iloc[:,1:])


# In[23]:


train_scaled = pd.DataFrame(train_scaled)


# In[24]:


train_scaled.head()


# In[25]:


train_scaled.describe()


# ## Doing PCA to get important variables only

# In[26]:


from sklearn.decomposition import PCA


# In[27]:


pca = PCA()


# In[28]:


pca.fit(train_scaled.iloc[:,1:])


# In[29]:


varience = pca.explained_variance_ratio_
varience.shape


# In[30]:


cum_var = np.cumsum(varience)
plt.plot(cum_var)


# #### We can see from the graph that we have to select at least 225 variables in order to avoid information loss

# In[31]:


pca1 = PCA(n_components=225)


# In[32]:


p_comp = pca1.fit_transform(train_scaled.iloc[:,1:])


# In[33]:


x = pd.DataFrame(p_comp)


# In[34]:


y = train_scaled[0]


# In[35]:


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.3)   ## Splitting the data into train and test sets


# In[36]:


from sklearn.linear_model import LinearRegression    ## Fitting the linear regression model
lr = LinearRegression()


# In[37]:


lr.fit(x_train,y_train)


# In[38]:


y_pred = lr.predict(x_test)


# In[39]:


from sklearn.metrics import r2_score,mean_absolute_error,mean_squared_error


# In[40]:


r2 = r2_score(y_test,y_pred)
r2


# In[41]:


MAE = mean_absolute_error(y_test,y_pred)
MAE


# In[42]:


MSE = mean_squared_error(y_test,y_pred)
MSE


# In[43]:


RMSE = np.sqrt(MSE)
RMSE


# In[44]:


test = pd.read_csv('test.csv')


# In[45]:


test.head()


# In[46]:


test.var()[test.var()==0]    ## Finding the variables having zero varience


# In[47]:


test_new = test.drop(test.var()[test.var()==0].index.values,axis=1)    ## dropping the variables having zero varience


# In[48]:


test_new.size


# In[49]:


test_new.shape


# In[50]:


test_new.isnull().sum()


# In[51]:


test_new.dropna()


# In[52]:


test_new.info()


# In[53]:


test_new.select_dtypes(object)


# In[54]:


columns1 = ['X0','X1','X2','X3','X4','X5','X6','X8']


# In[55]:


test_new[columns1] = test_new[columns1].apply(le.fit_transform)
test_new.head()


# In[56]:


test_scaled = sc.fit_transform(test_new)


# In[57]:


test_scaled = pd.DataFrame(test_scaled)


# In[58]:


test_scaled.head()


# In[59]:


test_scaled.describe()


# In[60]:


from sklearn.model_selection import KFold,cross_val_score
from sklearn.ensemble import GradientBoostingRegressor


# In[61]:


xgb = GradientBoostingRegressor(n_estimators=30)


# In[62]:


model = xgb.fit(x_train,y_train)


# In[63]:


y_pred1 = xgb.predict(x_test)


# In[64]:


kfold = KFold(n_splits=10)
result = cross_val_score(model,x_test,y_test,cv=kfold)    ## Doing cross validation


# In[65]:


result.mean()


# In[66]:


pcomp1 = pca1.fit_transform(test_scaled)

x_features = pd.DataFrame(pcomp1)


# In[67]:


Final_Predictions = xgb.predict(x_features)


# In[68]:


Final_Predictions = pd.DataFrame(Final_Predictions)


# In[69]:


Final_Predictions

