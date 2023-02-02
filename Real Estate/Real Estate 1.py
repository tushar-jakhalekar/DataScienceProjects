#!/usr/bin/env python
# coding: utf-8

# # Real Estate
# 
# #### A banking institution requires actionable insights from the perspective of Mortgage-Backed Securities, Geographic Business Investment and Real Estate Analysis. 
# #### The objective is to identify white spaces/potential business in the mortgage loan. The mortgage bank would like to identify potential monthly mortgage expenses for each of region based on factors which are primarily monthly family income in a region and rented value of the real estate. Some of the regions are growing rapidly and Competitor banks are selling mortgage loans to subprime customers at a lower interest rate. The bank is strategizing for better market penetration and targeting new customers. A statistical model needs to be created to predict the potential demand in dollars amount of loan for each of the region in the USA. Also, there is a need to create a dashboard which would refresh periodically post data retrieval from the agencies. This would help to monitor the key metrics and trends.

# ### Import the required libraries

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import warnings
warnings.filterwarnings('ignore')
import seaborn as sns


# ### Import the data

# In[2]:


df = pd.read_csv('T:\Masters In Data Science\Capstone Project\Project 1\\train.csv')


# In[3]:


df.head()


# In[4]:


df.shape


# In[5]:


df.info()   ## checking for null values in the data as well as data types of several variables


# #### Null values treatment

# In[6]:


df.isnull().sum()


# In[7]:


df_train = df.drop('BLOCKID',axis=1)


# In[8]:


df_train.head()


# In[9]:


df_train.isnull().sum()


# In[10]:


df_train.dropna(inplace=True)   ## Dropping the null values


# In[11]:


df_train.isnull().sum()


# ### We are taking the top 2500 locations where Second Mortgage is highest and Percentage Ownership is also above 10%

# In[12]:


df_train1 = df_train.nlargest(2500,['second_mortgage','pct_own'])


# In[13]:


df_train1.shape


# In[14]:


df_train1.head()


# In[15]:


df_train1['Bad_debt'] = df_train1['second_mortgage']+df_train1['home_equity']-df_train1['home_equity_second_mortgage']


# In[16]:


df_train1.head()


# In[17]:


df_train1['Good_debt'] = df_train1['debt']-df_train1['Bad_debt']


# In[18]:


df_train1.head()


# In[19]:


df_train1.describe()


# In[20]:


piechart = df_train1[['place','debt','Bad_debt','Good_debt']].reset_index()


# In[21]:


piechart.head()


# In[22]:


l1 = list(piechart['Bad_debt'])
l1[:10]


# In[23]:


l2 = list(piechart['Good_debt'])
l2[:10]


# In[24]:


l3 = sum(zip(l1,l2+[0]),())
l3[:20]


# In[25]:


debt_good_bad = l3[:20]

size = 10
labels_D = ['GD', 'BD'] * size
labels_D = tuple(labels_D)
labels_D


# In[26]:


color_pal = plt.rcParams['axes.prop_cycle'].by_key()['color']


# In[27]:


sns.set_style("whitegrid")

plt.figure(figsize = (10,10))

plt.pie(piechart.debt[:10], labels=piechart.place[:10],autopct = '%0.2f%%',radius=25,startangle = 90,pctdistance=0.85, labeldistance = 0.9, frame = True,colors = color_pal)
plt.pie(debt_good_bad[:20],labels =labels_D ,autopct = '%0.2f%%',radius=20,startangle = 90,pctdistance=0.85, labeldistance = 0.9, frame = True,colors = color_pal)
center_circle = plt.Circle((0,0),10,color='black', fc='white',linewidth=0)
fig = plt.gcf()
fig.gca().add_artist(center_circle)
plt.axis('equal')
plt.title('Debt Analysis',fontsize=20)
plt.tight_layout()
plt.show()


# #### Pie chart shows the Overall debt and Good debt and Bad debt as part of overall debt for top 10 cities
# #### Here we can see that Millbourne is having maximum debt percentage out of top 10 cities and 8.49% of the debt is good debt for the city 

# In[28]:


city_list = df_train1['city'].value_counts()[:30].index


# In[29]:


city_list


# In[30]:


boxplot = df_train1[df_train1['city'].isin(city_list)]


# In[31]:


sns.set_style('whitegrid')

plt.figure(figsize = (35,10))
sns.boxplot(x='city',y='second_mortgage',data=boxplot,palette='rainbow',order=['Chicago', 'Los Angeles', 'Washington', 'Brooklyn', 'Milwaukee','Aurora', 'Jacksonville', 'Las Vegas', 'Denver', 'Charlotte', 'Bronx','Baltimore', 'Long Beach', 'Minneapolis', 'Colorado Springs','New Orleans', 'Cincinnati', 'Sacramento', 'Columbus', 'San Diego','Lowell', 'Dallas', 'Atlanta', 'Alexandria', 'Orlando', 'Oakland','Miami', 'San Jose', 'Portland', 'Littleton'])
plt.title('Second Mortgage distribution by cities',fontsize=20)
plt.show()


# In[32]:


sns.set_style('whitegrid')

plt.figure(figsize = (35,10))
sns.boxplot(x='city',y='home_equity',data=boxplot,palette='rainbow',order=['Chicago', 'Los Angeles', 'Washington', 'Brooklyn', 'Milwaukee','Aurora', 'Jacksonville', 'Las Vegas', 'Denver', 'Charlotte', 'Bronx','Baltimore', 'Long Beach', 'Minneapolis', 'Colorado Springs','New Orleans', 'Cincinnati', 'Sacramento', 'Columbus', 'San Diego','Lowell', 'Dallas', 'Atlanta', 'Alexandria', 'Orlando', 'Oakland','Miami', 'San Jose', 'Portland', 'Littleton'])
plt.title('Home Equity distribution by cities',fontsize=20)
plt.show()


# In[33]:


sns.set_style('whitegrid')

plt.figure(figsize = (35,10))
sns.boxplot(x='city',y='Good_debt',data=boxplot,palette='rainbow',order=['Chicago', 'Los Angeles', 'Washington', 'Brooklyn', 'Milwaukee','Aurora', 'Jacksonville', 'Las Vegas', 'Denver', 'Charlotte', 'Bronx','Baltimore', 'Long Beach', 'Minneapolis', 'Colorado Springs','New Orleans', 'Cincinnati', 'Sacramento', 'Columbus', 'San Diego','Lowell', 'Dallas', 'Atlanta', 'Alexandria', 'Orlando', 'Oakland','Miami', 'San Jose', 'Portland', 'Littleton'])
plt.title('Good Debt distribution by cities',fontsize=20)
plt.show()


# In[34]:


sns.set_style('whitegrid')

plt.figure(figsize = (35,10))
sns.boxplot(x='city',y='Bad_debt',data=boxplot,palette='rainbow',order=['Chicago', 'Los Angeles', 'Washington', 'Brooklyn', 'Milwaukee','Aurora', 'Jacksonville', 'Las Vegas', 'Denver', 'Charlotte', 'Bronx','Baltimore', 'Long Beach', 'Minneapolis', 'Colorado Springs','New Orleans', 'Cincinnati', 'Sacramento', 'Columbus', 'San Diego','Lowell', 'Dallas', 'Atlanta', 'Alexandria', 'Orlando', 'Oakland','Miami', 'San Jose', 'Portland', 'Littleton'])
plt.title('Bad Debt distribution by cities',fontsize=20)
plt.show()


# In[35]:


df_train1['remaining_income'] = df_train1['family_median']-df_train1['hi_median']


# In[36]:


sns.set_style('whitegrid')

plt.figure(figsize = (5,5))
sns.boxplot(data=df_train1[['hi_median','family_median','remaining_income']],palette=color_pal)
plt.title('Collated distribution chart',fontsize=20)
plt.show()


# In[37]:


plt.figure(figsize=(10,10))
sns.histplot(df_train1.hi_median,kde=True,bins=20,color='y',label='hi_median')
sns.histplot(df_train1.family_median,kde=True,bins=20,color='r',label='family_median')
sns.histplot(df_train1.remaining_income,kde=True,bins=20,color='b',label='remaining_income')
plt.legend()
plt.title('Collated Distribution Chart',fontsize=20)
plt.show()


# In[38]:


df_train1['Population_density'] = df_train1['pop'] / df_train1['ALand']


# In[39]:


df_train1.head()


# In[40]:


pop_density_gb = df_train1.groupby('state')['Population_density'].sum().reset_index()


# In[41]:


plt.figure(figsize = (14,14))
sns.barplot(x = 'state', y = 'Population_density',data = pop_density_gb,orient='v').set_xticklabels(pop_density_gb['state'].values,rotation=90)
plt.title('State-wise Population density chart',fontsize=20)
plt.show()


# #### The barplot shows the citywise population density
# #### California and New York are more densely populated than other cities where as South Dakota is least densly populated

# In[42]:


df_train1['median_age'] = (df_train1['male_age_median']*df_train1['male_pop'])+(df_train1['female_age_median']*df_train1['female_pop']) / df_train1['pop']


# In[43]:


df_train1.head()


# In[44]:


df_med_age = df_train1.groupby('state')['median_age'].size().reset_index()


# In[45]:


df_med_age.head()


# In[46]:


plt.figure(figsize = (14,14))
sns.barplot(x='state',y='median_age',data=df_med_age).set_xticklabels(df_med_age['state'].values,rotation=90)
plt.title('State-wise median age chart',fontsize=20)
plt.show()


# #### California has highest median age as compared to other cities which means there are more elderly people living in california than other cities

# In[47]:


df_train1.columns


# In[48]:


df_for_age_analysis = df_train1[['state','city','place','pop','male_pop','female_pop','male_age_median','female_age_median','married','separated','divorced']]


# In[49]:


df_for_age_analysis['male_age_median'].unique()


# In[50]:


df_for_age_analysis['male_pop_labels'] = pd.cut(df_for_age_analysis['male_age_median'],bins=[0,10,18,40,60,100],labels=['Kids','Youth','Adult_youth','Adult','Senior_citizen'])


# In[51]:


df_for_age_analysis['female_pop_labels'] = pd.cut(df_for_age_analysis['female_age_median'],bins=[0,10,18,40,60,100],labels=['Kids','Youth','Adult_youth','Adult','Senior_citizen'])


# In[52]:


df_for_age_analysis['state'].value_counts()[:30].index


# In[53]:


plt.figure(figsize=(35,10))
sns.barplot(x='state',y='married',data=df_for_age_analysis,hue='male_pop_labels',order=['California', 'Colorado', 'Florida', 'Georgia', 'New York', 'Virginia',
       'Ohio', 'Maryland', 'Illinois', 'Minnesota', 'Massachusetts', 'Texas',
       'Washington', 'Michigan', 'Connecticut', 'North Carolina', 'Wisconsin',
       'Oregon', 'Utah', 'New Jersey', 'Pennsylvania', 'Nevada', 'Arizona',
       'Louisiana', 'Missouri', 'Tennessee', 'District of Columbia', 'Indiana',
       'Kentucky', 'South Carolina'])
plt.title('Statewise married male population',fontsize=20)
plt.show()


# In[54]:


plt.figure(figsize=(35,10))
sns.barplot(x='state',y='separated',data=df_for_age_analysis,hue='male_pop_labels',order=['California', 'Colorado', 'Florida', 'Georgia', 'New York', 'Virginia',
       'Ohio', 'Maryland', 'Illinois', 'Minnesota', 'Massachusetts', 'Texas',
       'Washington', 'Michigan', 'Connecticut', 'North Carolina', 'Wisconsin',
       'Oregon', 'Utah', 'New Jersey', 'Pennsylvania', 'Nevada', 'Arizona',
       'Louisiana', 'Missouri', 'Tennessee', 'District of Columbia', 'Indiana',
       'Kentucky', 'South Carolina'])
plt.title('Statewise Separated male population',fontsize=20)
plt.show()


# In[55]:


plt.figure(figsize=(35,10))
sns.barplot(x='state',y='divorced',data=df_for_age_analysis,hue='male_pop_labels',order=['California', 'Colorado', 'Florida', 'Georgia', 'New York', 'Virginia',
       'Ohio', 'Maryland', 'Illinois', 'Minnesota', 'Massachusetts', 'Texas',
       'Washington', 'Michigan', 'Connecticut', 'North Carolina', 'Wisconsin',
       'Oregon', 'Utah', 'New Jersey', 'Pennsylvania', 'Nevada', 'Arizona',
       'Louisiana', 'Missouri', 'Tennessee', 'District of Columbia', 'Indiana',
       'Kentucky', 'South Carolina'])
plt.title('Statewise Divorced male population',fontsize=20)
plt.show()


# In[56]:


plt.figure(figsize=(35,10))
sns.barplot(x='state',y='married',data=df_for_age_analysis,hue='female_pop_labels',order=['California', 'Colorado', 'Florida', 'Georgia', 'New York', 'Virginia',
       'Ohio', 'Maryland', 'Illinois', 'Minnesota', 'Massachusetts', 'Texas',
       'Washington', 'Michigan', 'Connecticut', 'North Carolina', 'Wisconsin',
       'Oregon', 'Utah', 'New Jersey', 'Pennsylvania', 'Nevada', 'Arizona',
       'Louisiana', 'Missouri', 'Tennessee', 'District of Columbia', 'Indiana',
       'Kentucky', 'South Carolina'])
plt.title('Statewise married female population',fontsize=20)
plt.show()


# In[57]:


plt.figure(figsize=(35,10))
sns.barplot(x='state',y='separated',data=df_for_age_analysis,hue='female_pop_labels',order=['California', 'Colorado', 'Florida', 'Georgia', 'New York', 'Virginia',
       'Ohio', 'Maryland', 'Illinois', 'Minnesota', 'Massachusetts', 'Texas',
       'Washington', 'Michigan', 'Connecticut', 'North Carolina', 'Wisconsin',
       'Oregon', 'Utah', 'New Jersey', 'Pennsylvania', 'Nevada', 'Arizona',
       'Louisiana', 'Missouri', 'Tennessee', 'District of Columbia', 'Indiana',
       'Kentucky', 'South Carolina'])
plt.title('Statewise Separated female population',fontsize=20)
plt.show()


# In[58]:


plt.figure(figsize=(35,10))
sns.barplot(x='state',y='divorced',data=df_for_age_analysis,hue='female_pop_labels',order=['California', 'Colorado', 'Florida', 'Georgia', 'New York', 'Virginia',
       'Ohio', 'Maryland', 'Illinois', 'Minnesota', 'Massachusetts', 'Texas',
       'Washington', 'Michigan', 'Connecticut', 'North Carolina', 'Wisconsin',
       'Oregon', 'Utah', 'New Jersey', 'Pennsylvania', 'Nevada', 'Arizona',
       'Louisiana', 'Missouri', 'Tennessee', 'District of Columbia', 'Indiana',
       'Kentucky', 'South Carolina'])
plt.title('Statewise Divorced female population',fontsize=20)
plt.show()


# In[59]:


round(df_train1['rent_median'].sum()/df_train1['hi_median'].sum()*100,2)


# In[60]:


df_train1['rent%'] = round(df_train1['rent_median']/df_train1['hi_median']*100,2)


# In[61]:


df_train1.head()


# In[62]:


rent_df = df_train1.groupby('state')['rent%'].median().reset_index()


# In[63]:


rent_df.head()


# In[64]:


plt.figure(figsize=(14,14))
sns.barplot(x='state',y='rent%',data=rent_df,palette='tab10').set_xticklabels(rent_df['state'].values,rotation=90)
plt.title('Statewise rent as % of overall income',fontsize=20)
plt.show()


# #### People from Puerto Rico are having less income and paying most rent as percentage of their income where as South Dakota people are having less rent% as their income

# In[65]:


corr = df_train1.corr()


# In[66]:


positive_correlation = corr[corr>=0]
negative_correlation = corr[corr<0]


# In[67]:


plt.figure(figsize = (45,30))
sns.heatmap(positive_correlation,cmap='Greens',annot=True,linecolor='red',linewidths=1)
plt.title('Positive Correlation Heatmap',fontsize=40)
plt.show()


# In[68]:


plt.figure(figsize = (45,30))
sns.heatmap(negative_correlation,cmap='Blues',annot=True,linecolor='red',linewidths=1)
plt.title('Negative Correlation Heatmap',fontsize=40)
plt.show()


# ## Data Preprocessing

# In[69]:


df_train1.describe()


# In[70]:


df_train1.info()


# In[71]:


numerical_variables = df_train1.select_dtypes(('int64','float64'))


# In[72]:


numerical_variables.shape


# In[73]:


numerical_variables.drop(['SUMLEVEL','lat','lng','ALand','AWater'],axis=1,inplace=True)


# In[74]:


numerical_variables.shape


# In[75]:


from sklearn.decomposition import FactorAnalysis
fa = FactorAnalysis(n_components=25)


# In[76]:


fact = fa.fit_transform(numerical_variables)


# In[77]:


fact


# In[78]:


plt.scatter(fact[:,0],fact[:,1])


# In[79]:


variables = pd.DataFrame(fact)


# In[80]:


variables.head()


# In[81]:


numerical_variables.isnull().sum()


# In[82]:


numerical_variables['hc_mortgage_mean'].isnull().sum()


# In[83]:


numerical_variables.columns


# In[84]:


x = numerical_variables[['UID', 'COUNTYID', 'STATEID', 'zip_code', 'area_code', 'pop',
       'male_pop', 'female_pop', 'rent_mean', 'rent_median', 'rent_stdev',
       'rent_sample_weight', 'rent_samples', 'rent_gt_10', 'rent_gt_15',
       'rent_gt_20', 'rent_gt_25', 'rent_gt_30', 'rent_gt_35', 'rent_gt_40',
       'rent_gt_50', 'universe_samples', 'used_samples', 'hi_mean',
       'hi_median', 'hi_stdev', 'hi_sample_weight', 'hi_samples',
       'family_mean', 'family_median', 'family_stdev', 'family_sample_weight',
       'family_samples', 'hc_mortgage_median',
       'hc_mortgage_stdev', 'hc_mortgage_sample_weight', 'hc_mortgage_samples',
       'hc_mean', 'hc_median', 'hc_stdev', 'hc_samples', 'hc_sample_weight',
       'home_equity_second_mortgage', 'second_mortgage', 'home_equity', 'debt',
       'second_mortgage_cdf', 'home_equity_cdf', 'debt_cdf', 'hs_degree',
       'hs_degree_male', 'hs_degree_female', 'male_age_mean',
       'male_age_median', 'male_age_stdev', 'male_age_sample_weight',
       'male_age_samples', 'female_age_mean', 'female_age_median',
       'female_age_stdev', 'female_age_sample_weight', 'female_age_samples',
       'pct_own', 'married', 'married_snp', 'separated', 'divorced',
       'Bad_debt', 'Good_debt', 'remaining_income', 'Population_density',
       'median_age', 'rent%']]
y = numerical_variables['hc_mortgage_mean']


# #### Splitting the data as train and test 

# In[85]:


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.3,random_state=50)


# In[86]:


x_train.shape


# In[87]:


x_test.shape


# In[88]:


y_train.shape


# In[89]:


y_test.shape


# #### Here we are using multi-linear regression model

# In[90]:


from sklearn.linear_model import LinearRegression
lm = LinearRegression()


# In[91]:


Model = lm.fit(x_train,y_train)


# In[92]:


y_pred = lm.predict(x_test)


# In[93]:


from sklearn.metrics import r2_score,mean_absolute_error,mean_squared_error


# In[94]:


MAE = mean_absolute_error(y_test,y_pred)
MAE


# In[95]:


MSE = mean_squared_error(y_test,y_pred)
MSE


# In[96]:


RMSE = np.sqrt(MSE)
RMSE


# In[97]:


r2 = r2_score(y_test,y_pred)
r2


# #### We got 98.27% r2 score which is above acceptance limit so we can skip the remaining steps now we have to predict the valuse for hc_mortgage_mean for the test dataset

# In[98]:


df_test = pd.read_csv('T:\Masters In Data Science\Capstone Project\Project 1\\test.csv')


# In[99]:


df_test.head()


# In[100]:


df_test.info()


# In[101]:


df_test.isnull().sum()


# In[102]:


df_test = df_test.drop(['BLOCKID'],axis=1)


# In[103]:


df_test.isnull().sum()


# In[104]:


df_test.dropna(inplace=True)


# In[105]:


df_test.info()


# In[106]:


df_test1 = df_test.nlargest(2500,['second_mortgage','pct_own'])


# In[107]:


df_test1.shape


# In[108]:


df_test1['Bad_debt'] = df_test1['second_mortgage'] + df_test1['home_equity'] - df_test1['home_equity_second_mortgage']
df_test1['Good_debt'] = df_test1['debt'] - df_test1['Bad_debt']


# In[109]:


df_test1.describe()


# In[110]:


df_test1['Remaining_income'] = df_test1['family_median'] - df_test1['hi_median']


# In[111]:


df_test1['Population_density'] = df_test1['pop'] / df_test1['ALand']


# In[112]:


df_test1['median_age'] = (df_test1['male_age_median']*df_test1['male_pop'])+(df_test1['female_age_median']*df_test1['female_pop']) / df_test1['pop']


# In[113]:


df_test1['rent%'] = round(df_test1['rent_median']/df_test1['hi_median']*100,2)


# In[114]:


df_test1.head()


# In[115]:


df_test1.info()


# In[116]:


numerical_variables_test = df_test1.select_dtypes(('int64','float64'))


# In[117]:


numerical_variables_test.head()


# In[118]:


numerical_variables_test.drop(['SUMLEVEL','lat','lng','ALand','AWater'],axis=1,inplace=True)


# In[119]:


numerical_variables_test.shape


# In[120]:


numerical_variables_test.columns


# In[121]:


X = numerical_variables_test[['UID', 'COUNTYID', 'STATEID', 'zip_code', 'area_code', 'pop',
       'male_pop', 'female_pop', 'rent_mean', 'rent_median', 'rent_stdev',
       'rent_sample_weight', 'rent_samples', 'rent_gt_10', 'rent_gt_15',
       'rent_gt_20', 'rent_gt_25', 'rent_gt_30', 'rent_gt_35', 'rent_gt_40',
       'rent_gt_50', 'universe_samples', 'used_samples', 'hi_mean',
       'hi_median', 'hi_stdev', 'hi_sample_weight', 'hi_samples',
       'family_mean', 'family_median', 'family_stdev', 'family_sample_weight',
       'family_samples', 'hc_mortgage_median',
       'hc_mortgage_stdev', 'hc_mortgage_sample_weight', 'hc_mortgage_samples',
       'hc_mean', 'hc_median', 'hc_stdev', 'hc_samples', 'hc_sample_weight',
       'home_equity_second_mortgage', 'second_mortgage', 'home_equity', 'debt',
       'second_mortgage_cdf', 'home_equity_cdf', 'debt_cdf', 'hs_degree',
       'hs_degree_male', 'hs_degree_female', 'male_age_mean',
       'male_age_median', 'male_age_stdev', 'male_age_sample_weight',
       'male_age_samples', 'female_age_mean', 'female_age_median',
       'female_age_stdev', 'female_age_sample_weight', 'female_age_samples',
       'pct_own', 'married', 'married_snp', 'separated', 'divorced',
       'Bad_debt', 'Good_debt', 'Remaining_income', 'Population_density',
       'median_age', 'rent%']]
Y = numerical_variables_test['hc_mortgage_mean']


# In[122]:


Y_Pred = lm.predict(X)


# In[123]:


MAE1 = mean_absolute_error(Y,Y_Pred)
MAE1


# In[124]:


MSE1 = mean_squared_error(Y,Y_Pred)
MSE1


# In[125]:


RMSE1 = np.sqrt(MSE1)
RMSE1


# In[126]:


r2_1 = r2_score(Y,Y_Pred)
r2_1


# #### Here we have 98.75% r2 score so we can skip the state level model building procedure

# In[127]:


Check = pd.DataFrame({'Predicted hc_mortgage_mean' : Y_Pred , 'Actual hc_mortgage_mean' : Y})
Check


# In[128]:


from statsmodels.stats.outliers_influence import variance_inflation_factor


# In[129]:


VIF_data = pd.DataFrame()


# In[130]:


VIF_data['features'] = numerical_variables_test.columns


# In[131]:


VIF_data['VIF'] = [variance_inflation_factor(numerical_variables_test.values,i)
                  for i in range (len(numerical_variables_test.columns))]


# In[132]:


print(VIF_data)


# In[133]:


plt.figure(figsize=(15,15))
sns.histplot(data=Y_Pred,color='c',bins=20,kde=True)
plt.title('Predicted Values Distribution')
plt.show()


# #### The predicted data looks somewhat right skewed

# #### Now we will use pandas function to extract the top 2500 dataframe into csv format for Dashboarding use

# In[134]:


df_train1.to_csv('Real_Estate.csv')


# In[136]:


df_train.corr().to_csv('Correlation.csv')


# In[ ]:




