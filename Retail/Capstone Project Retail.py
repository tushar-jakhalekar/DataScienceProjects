#!/usr/bin/env python
# coding: utf-8

# ## Retail 
# 
# #### It is a business critical requirement to understand the value derived from a customer. RFM is a method used for analyzing customer value.
# #### Perform customer segmentation using RFM analysis. The resulting segments can be ordered from most valuable (highest recency, frequency, and value) to least valuable (lowest recency, frequency, and value). Identifying the most valuable RFM segments can capitalize on chance relationships in the data used for this analysis
# 

# In[1]:


import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')
import matplotlib.pyplot as plt


# In[2]:


df = pd.read_excel('T:\Masters In Data Science\Capstone Project\Project 3\\Online Retail.xlsx')


# In[3]:


df.head()


# In[4]:


df.shape


# ### Descriptive Analysis

# In[5]:


df.describe()


# #### Unit Price : Average Unit price sold id 4.6 Also we have to note that Min unit price is negative which means store had to return some amount for returned products during the period of our analysis
# #### Quantity : Average quantity bought is 9.55 Also some products were returned to the store by customers during the period of our analysis

# In[6]:


df.describe(include='O')


# #### Invoice No : The total number of invoices created is 25900
# #### Country : Store is functional in 38 countries
# #### Stock Code : There are total 4070 types of items in stock 

# In[7]:


df.info()


# ### Dropping the duplicates

# In[8]:


df = df.drop_duplicates()


# In[9]:


df.info()


# In[10]:


df.isnull().sum()


# In[11]:


round((df.isnull().sum()/len(df))*100,2)


# #### There are 25.16% null values in the column Customer Id we will see if we can find any customer Ids in the invoices otherwise we will drop them from the dataset

# In[12]:


null_cust_id_inv = set(df[df['CustomerID'].isnull()]['InvoiceNo'])


# In[13]:


df[df['InvoiceNo'].isin(null_cust_id_inv) & (~df['CustomerID'].isnull())]


# #### We could not find any customer Ids using invoice numbers so we will drop the null rows from the dataset 

# In[14]:


df = df.dropna()
df.shape


# ### Cohort Analysis
# #### a.	Create month cohorts and analyse active  customers for each cohort

# In[15]:


from datetime import timedelta
df['Month_of_year'] = df['InvoiceDate'].dt.to_period('M')
df['Month_of_year'].nunique()


# In[16]:


Month_Cohort = df.groupby('Month_of_year')['CustomerID'].nunique()
Month_Cohort


# In[17]:


import seaborn as sns


# In[18]:


plt.figure(figsize=(10,7))
sns.barplot(x = Month_Cohort.values, y = Month_Cohort.index)
plt.xlabel('Number of Customers')
plt.title('Per month active customers',fontsize=14)
plt.show()


# In[19]:


Month_Cohort = Month_Cohort.shift(1)


# #### b) Analyse the retention rate of customers. 

# In[20]:


Retention_rate = round(Month_Cohort.pct_change(periods=1)*100,2)
Retention_rate


# In[21]:


plt.figure(figsize=(10,7))
sns.barplot(x=Retention_rate.values,y=Retention_rate.index)
plt.xlabel('Retention Rate In %')
plt.title('Monthwise Customer retention rate')
plt.show()


# #### Monetary analysis

# In[22]:


df['amount'] = df['Quantity'] * df['UnitPrice']


# In[23]:


monetary = df.groupby('CustomerID')['amount'].sum().reset_index()


# In[24]:


monetary


# #### Frequency analysis

# In[25]:


df.head()


# In[26]:


Frequency = df.groupby('CustomerID').nunique()['InvoiceNo'].reset_index()
Frequency


# #### Recency analysis

# In[27]:


ref_day = max(df['InvoiceDate']) + timedelta(days=1)
ref_day


# In[28]:


df['days_since_last_order'] = (ref_day - df['InvoiceDate'])


# In[29]:


Recency = df.groupby('CustomerID').nunique()['days_since_last_order'].reset_index()
Recency


# In[30]:


RFM = pd.merge(Recency,Frequency,on='CustomerID',how='left')
RFM = pd.merge(RFM,monetary,on='CustomerID',how='left')


# In[31]:


RFM


# In[32]:


RFM.columns = ['Customer_id','Recency','Frequency','Monetary']


# In[33]:


RFM


# In[34]:


RFM['recency_labels'] = pd.cut(RFM['Recency'],bins=5,labels=['newest','newer','medium','older','oldest'])
RFM['recency_labels'].value_counts().plot(kind='barh')
RFM['recency_labels'].value_counts()


# In[35]:


RFM['frequency_labels'] = pd.cut(RFM['Frequency'],bins=5,labels=['lowest','lower','medium','higher','highest'])
RFM['frequency_labels'].value_counts().plot(kind='barh')
RFM['frequency_labels'].value_counts()


# In[36]:


RFM['monetary_labels'] = pd.cut(RFM['Monetary'],bins=5,labels=['smallest','smaller','medium','larger','largest'])
RFM['monetary_labels'].value_counts().plot(kind='barh')
RFM['monetary_labels'].value_counts()


# #### RFM Segment

# In[37]:


RFM['Segment'] = RFM[['recency_labels','frequency_labels','monetary_labels']].agg('-'.join,axis=1)
RFM.head()


# #### RFM score

# In[38]:


recency_dict = {'newest':1,'newer':2,'medium':3,'older':4,'oldest':5}
frequency_dict = {'lowest':1,'lower':2,'medium':3,'higher':4,'highest':5}
monetary_dict = {'smallest':1,'smaller':2,'medium':3,'larger':4,'largest':5}


# In[39]:


RFM['rfm_score'] = RFM['recency_labels'].map(recency_dict).astype(int) + RFM['frequency_labels'].map(frequency_dict).astype(int)+ RFM['monetary_labels'].map(monetary_dict).astype(int)


# In[40]:


RFM.head()


# In[41]:


plt.figure(figsize=(8,8))
RFM['rfm_score'].value_counts().plot(kind='barh')
RFM['rfm_score'].value_counts()


# #### This shows that most of the customers are new in the store also they visit the store less frequently and they spend less amount of money in the store

# ### Data Preprocessing

# In[42]:


RFM.shape


# In[43]:


RFM.head()


# #### Let us check the distribution of the data

# In[44]:


plt.figure(figsize=(20,10))
RFM[['Recency','Frequency','Monetary']].hist(bins=6)
plt.title('Histplot of Recency, Frequency, Monetary metrics',fontsize=14)
plt.show()


# In[45]:


plt.figure(figsize = (20,10))
sns.set_style('whitegrid')
sns.boxplot(data=RFM[['Recency','Frequency','Monetary']],palette='rainbow')
plt.title('Boxplot for Recency, Frequency, Monetary metrics',fontsize=14)
plt.show()


# #### Looks like monetary metrics have a lot of outliers

# In[46]:


Q1 = RFM['Monetary'].quantile(0.25)
Q3 = RFM['Monetary'].quantile(0.75)
print('Q1=',Q1,'Q3=',Q3)

IQR = Q3-Q1
print(IQR)

Upper_whisker = Q3+1.5*IQR
Lower_whisker = Q1-1.5*IQR

print('Upper Whisker=',Upper_whisker)
print('Lower Whisker=',Lower_whisker)


# In[47]:


RFM_new = RFM[RFM['Monetary']<Upper_whisker]
RFM_new.shape


# In[48]:


plt.figure(figsize = (20,10))
sns.set_style('whitegrid')
sns.boxplot(data=RFM_new[['Recency','Frequency','Monetary']],palette='rainbow')
plt.title('New Boxplot for Recency, Frequency, Monetary metrics',fontsize=14)
plt.show()            


# #### Now let us scale the data for better use

# In[49]:


RFM_new_for_scaling = RFM_new[['Recency','Frequency','Monetary']]


# In[50]:


from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
RFM_scaled = scaler.fit_transform(RFM_new_for_scaling)


# In[51]:


RFM_scaled = pd.DataFrame(RFM_scaled)
RFM_scaled


# In[52]:


RFM_scaled.columns = ['Recency','Frequency','Monetary']
RFM_scaled


# #### Build K means model

# In[53]:


from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=3,max_iter=50)
kmeans.fit(RFM_scaled)


# In[54]:


kmeans.labels_


# In[55]:


blank_list = []

for i in range(1,11):
    kmeans = KMeans(n_clusters=i)
    kmeans.fit(RFM_scaled)
    blank_list.append(kmeans.inertia_)

plt.plot(range(1,11),blank_list)
plt.title('Elbow Method')
plt.xlabel('Number of K means clusters')
plt.ylabel('List')
plt.show()


# #### From elbow plot we can see after 4 clusters line starts to stabalize so we can select optimum number of clusters as 4

# In[56]:


df_inertia = pd.DataFrame(list(zip(range(1,11),blank_list)))
df_inertia.columns = ['clusters','inertia']
df_inertia


# #### Building Kmeans model again for 4 numbers of clusters

# In[57]:


kmeans = KMeans(n_clusters=4,max_iter=50)


# In[58]:


kmeans.fit(RFM_scaled)


# In[59]:


RFM_scaled['Cluster_ID'] = kmeans.labels_
RFM_scaled.head()


# In[60]:


sns.boxplot(x = 'Cluster_ID',y = 'Recency',data=RFM_scaled)
plt.title('Recency Plot')
plt.show()


# In[61]:


sns.boxplot(x = 'Cluster_ID',y = 'Frequency',data=RFM_scaled)
plt.title('Frequency Plot')
plt.show()


# In[62]:


sns.boxplot(x = 'Cluster_ID',y = 'Monetary',data=RFM_scaled)
plt.title('Monetary Plot')
plt.show()


# #### Cluster ID 0 : Cluster ID 0 contains customers who are having lowest frequency lowest recency and least money spent in our store these customers are least important from our perspective.
# #### Cluster ID 1 : Cluster ID 1 contains customers who are having lower frequent, have spent medium amount of money and have not bought very recently there are more important than cluster 0 but not more important than other customers.
# #### Cluster ID 2 : These customers are having medium frequency to visit store, they have spent some good amount after visiting the store so these are more important for the store.
# #### Cluster ID 3 : These customers have bought most recently from the store, They have spent most money and also visit the store more frequently visiting the store so they are most important for the store.

# ### Converting the processed files for tableau dashboarding purpose

# In[63]:


df.to_csv('Master_data.csv')
RFM.to_csv('RFM_analysis.csv')
df_inertia.to_csv('Inertia.csv')


# In[64]:


product_descr = df[['StockCode','Description']]
product_descr = product_descr.drop_duplicates()
product_descr.to_csv('Product_description.csv')


# In[ ]:




