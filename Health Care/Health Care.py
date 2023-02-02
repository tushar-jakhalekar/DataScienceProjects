#!/usr/bin/env python
# coding: utf-8

# # Health Care Capstone Project
# #### The objective of the dataset is to diagnostically predict whether or not a patient has diabetes, based on certain diagnostic measurements included in the dataset. Several constraints were placed on the selection of these instances from a larger database. In particular, all patients here are females at least 21 years old of Pima Indian heritage.

# #### Importing all the required libraries

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')


# In[2]:


df = pd.read_csv('T:\Masters In Data Science\Capstone Project\Project 2\Healthcare - Diabetes\health care diabetes.csv')


# In[3]:


df.head()


# In[4]:


df.shape


# In[5]:


df.info()


# In[6]:


df.columns


# In[7]:


print('For Insulin',df[df['Insulin'].values == 0].count())

print('For Glucose',df[df['Glucose'].values == 0].count())

print('For SkinThickness',df[df['SkinThickness'].values == 0].count())

print('For Blood Pressure',df[df['BloodPressure'].values == 0].count())

print('For BMI',df[df['BMI'].values == 0].count())


# #### As told the values which are 0 are actually null values BMI, Blood_Pressure, Glucose have less amount of null values but Insulin and Skin Thickness variables have significantly more number of null values we have to treat them accordingly

# In[8]:


import seaborn as sns


# In[9]:


sns.histplot(data=df['Glucose'],palette='rainbow')
plt.title('Initial distribution of Glucose readings',fontsize=14)
plt.show()


# In[10]:


sns.histplot(data=df['BloodPressure'],color='r',palette='rainbow')
plt.title('Initial distribution of Bloodpressure readings',fontsize=14)
plt.show()


# In[11]:


sns.histplot(data=df['Insulin'],color='y',palette='rainbow')
plt.title('Initial distribution of Insulin readings',fontsize=14)
plt.show()


# In[12]:


sns.histplot(data=df['BMI'],color='g',palette='rainbow')
plt.title('Initial distribution of BMI readings',fontsize=14)
plt.show()


# In[13]:


sns.histplot(data=df['SkinThickness'],color='b',palette='rainbow')
plt.title('Initial distribution of Skin Thickness readings',fontsize=14)
plt.show()


# #### For all the variables replacing null values with median of that particular variable.

# In[14]:


df['Glucose']=np.where(df.Glucose==0,df.Glucose.median(),df.Glucose)


# In[15]:


df['BloodPressure']=np.where(df.BloodPressure==0,df.BloodPressure.median(),df.BloodPressure)


# In[16]:


df['BMI']=np.where(df.BMI==0,df.BMI.median(),df.BMI)


# In[17]:


df['SkinThickness']=np.where(df.SkinThickness==0,df.SkinThickness.median(),df.SkinThickness)


# In[18]:


df['Insulin']=np.where(df.Insulin==0,df.Insulin.median(),df.Insulin)


# In[19]:


df['Insulin'].head()


# In[20]:


df['SkinThickness'].head()


# In[21]:


df.describe()


# #### Again checking the distribution of variables

# In[22]:


sns.histplot(data=df['Glucose'],palette='rainbow')
plt.title('Final distribution of Glucose readings',fontsize=14)
plt.show()


# In[23]:


sns.histplot(data=df['BloodPressure'],color='r',palette='rainbow')
plt.title('Final distribution of Blood Pressure readings',fontsize=14)
plt.show()


# In[24]:


sns.histplot(data=df['Insulin'],color='y',palette='rainbow')
plt.title('Final distribution of Insulin readings',fontsize=14)
plt.show()


# In[25]:


sns.histplot(data=df['BMI'],color='g',palette='rainbow')
plt.title('Final distribution of BMI readings',fontsize=14)
plt.show()


# In[26]:


sns.histplot(data=df['SkinThickness'],color='b',palette='rainbow')
plt.title('Final distribution of Skin Thickness readings',fontsize=14)
plt.show()


# In[27]:


df.columns


# #### Create a count plot describing the data types and the count of variables. 

# In[28]:


plt.figure(figsize=(10,10))
sns.countplot(data=df).set_xticklabels((df.dtypes),rotation=90)
plt.title('Countplot of datatypes in the dataset',fontsize=14)
plt.show()


# In[29]:


df.head()


# In[30]:


df['SkinThickness'] = round(df.SkinThickness,2)


# In[31]:


df['Insulin'] = round(df.Insulin,2)


# #### Check the balance of the data by plotting the count of outcomes by their value

# In[32]:


balance = df['Outcome'].value_counts().reset_index()


# In[33]:


balance


# In[34]:


sns.barplot(x='index',y='Outcome',data=balance)
plt.title('Outcome balance check',fontsize=14)
plt.show()


# #### From above plot it is clear that the data is imbalanced 

# In[35]:


plt.figure(figsize=(12,12))
sns.heatmap(df.corr(),annot=True)
plt.title('Heatmap of Correlation')
plt.show()


# #### We can see that Skin thickness and BMI have high positive correlation also Age and pregnancies have high correlation

# #### Checking the highly positively correlated variables relation by scatterplot

# In[36]:


plt.figure(figsize=(10,10))
sns.scatterplot(x='Pregnancies',y='Age',data=df,color='r')
plt.title('Relationship between Pregnancies and Age',fontsize=14)
plt.show()


# #### we can see as age increases number of pregnancies also increases

# In[37]:


plt.figure(figsize=(10,10))
sns.scatterplot(x='Glucose',y='Insulin',data=df,color='y')
plt.title('Relationship between Glucose and Insulin',fontsize=14)
plt.show()


# #### Relationship between Glucose and Insuline is somewhat linear

# In[38]:


plt.figure(figsize=(10,10))
sns.scatterplot(x='BMI',y='SkinThickness',data=df,color='g')
plt.title('Relationship between Skinthickness and BMI',fontsize=14)
plt.show()


# #### Similarly relation between Skinthickness and BMI is also somewhat linear

# ## Data Preprocessing

# In[39]:


df.columns


# In[40]:


X = df[['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin',
       'BMI', 'DiabetesPedigreeFunction', 'Age']]
Y = df['Outcome']


# In[41]:


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(X,Y,random_state=500,test_size=0.3)


# #### KNN model

# In[42]:


from sklearn.neighbors import KNeighborsClassifier
KNN = KNeighborsClassifier(n_neighbors=27)


# In[43]:


KNN_model = KNN.fit(x_test,y_test)


# In[44]:


y_pred_KNN = KNN.predict(x_test)
y_pred_KNN


# In[45]:


from sklearn.metrics import classification_report


# In[46]:


print(classification_report(y_test,y_pred_KNN))


# #### KNN model is having 74% Accuracy
# 
# #### Now we will try Logistic Regression model

# In[47]:


from sklearn.metrics import accuracy_score


# In[48]:


from sklearn.linear_model import LogisticRegression
logm = LogisticRegression()


# In[49]:


Logistic_regression_model = logm.fit(x_train,y_train)


# In[50]:


y_pred_logm = logm.predict(x_test)


# In[51]:


y_pred_logm


# In[52]:


from sklearn.metrics import classification_report
print(classification_report(y_test,y_pred_logm))


# #### Logistic Regression model has 79% Accuracy
# 
# #### Now we will try Decision tree model

# In[53]:


from sklearn.tree import DecisionTreeClassifier
dc = DecisionTreeClassifier()


# In[54]:


Decision_tree_model = dc.fit(x_train,y_train)
y_pred_decision_tree = dc.predict(x_test)


# In[55]:


y_pred_decision_tree


# In[56]:


print(classification_report(y_test,y_pred_decision_tree))


# #### Decision tree model has 73% Accuracy
# 
# #### Now we will try Random Forest Model

# In[57]:


from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(n_estimators=800)


# In[58]:


Random_forest_model = rf.fit(x_train,y_train)
y_pred_random_forest = rf.predict(x_test)


# In[59]:


y_pred_random_forest


# In[60]:


print(classification_report(y_test,y_pred_random_forest))


# #### We have achieved accuracy of 78% with Random forest model which is more than KNN model

# In[61]:


from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
kf = KFold(n_splits=20,random_state=25,shuffle=True)


# In[62]:


scores = cross_val_score(rf,X,Y,scoring='accuracy',cv=kf,n_jobs=1)
scores


# In[63]:


print('Accuracy : %.3f(%.3f)'%(np.mean(scores),np.std(scores)))


# #### This much lower accuracy is not acceptable hence we will try scaling the data and deploy all models again 

# In[64]:


from sklearn.preprocessing import StandardScaler
sc = StandardScaler()


# In[65]:


X1=sc.fit_transform(X)


# In[66]:


X11 = pd.DataFrame(X1)


# In[67]:


X11.head()


# In[68]:


x1_train,x1_test,y1_train,y1_test = train_test_split(X11,Y,test_size=0.33)


# #### KNN model

# In[69]:


KNN_model_1 = KNN.fit(x1_train,y1_train)
y_pred_KNN_1 = KNN.predict(x1_test)
y_pred_KNN_1


# In[70]:


print(classification_report(y1_test,y_pred_KNN_1))


# #### KNN has 80% accuracy
# 
# #### Now we will try Logistic regression model

# In[71]:


Logistic_regression_model_1 = logm.fit(x1_train,y1_train)
y_pred_logistic_regression_1 = logm.predict(x1_test)
y_pred_logistic_regression_1


# In[72]:


print(classification_report(y1_test,y_pred_logistic_regression_1))


# #### Logistic regression has 80% accuracy which is more than KNN
# 
# #### Now we will try Decision tree model

# In[73]:


Decision_tree_model_1 = dc.fit(x1_train,y1_train)
y_pred_decision_tree_1 = dc.predict(x1_test)
y_pred_decision_tree_1


# In[74]:


print(classification_report(y1_test,y_pred_decision_tree_1))


# #### Decision tree has 68% accuracy which is less than KNN
# 
# #### Now we will try Random forest model

# In[75]:


Random_forest_model_1 = rf.fit(x1_train,y1_train)
y_pred_random_forest_1 = rf.predict(x1_test)
y_pred_random_forest_1


# In[76]:


print(classification_report(y1_test,y_pred_random_forest_1))


# #### random Forest has 78% accuracy which is less than KNN

# In[77]:


from sklearn.metrics import confusion_matrix,ConfusionMatrixDisplay


# In[78]:


conf = confusion_matrix(y1_test,y_pred_KNN_1)


# In[79]:


conf


# #### Confusion Matrix for model with highest accuracy (KNN with scaling)

# In[80]:


matrix = ConfusionMatrixDisplay(confusion_matrix=conf,display_labels=[True,False])
matrix.plot()


# In[81]:


conf1 = confusion_matrix(y1_test,y_pred_decision_tree_1)


# #### Confusion matrix for model with lowest accuracy (Decision Tree with Scaling)

# In[82]:


matrix1 = ConfusionMatrixDisplay(confusion_matrix=conf1,display_labels=[True,False])
matrix1.plot()


# #### Collecting the results in a single dataframe

# In[83]:


Results = [{'KNN_without_scaling': accuracy_score(y_test,y_pred_KNN),
          'Logistic_Regression_without_scaling': accuracy_score(y_test,y_pred_logm),
          'Decision_tree_without_scaling': accuracy_score(y_test,y_pred_decision_tree),
          'Random_forest_without_scaling': accuracy_score(y_test,y_pred_random_forest),
          '':'',
          'KNN_with_scaling': accuracy_score(y1_test,y_pred_KNN_1),
          'Logistic_Regression_with_scaling' : accuracy_score(y1_test,y_pred_logistic_regression_1),
          'Decision_tree_with_scaling' : accuracy_score(y1_test,y_pred_decision_tree_1),
          'Random_forest_with_scaling' : accuracy_score(y1_test,y_pred_random_forest_1)}]


# In[84]:


Result = pd.DataFrame.from_dict(Results)


# In[85]:


Result=np.transpose(Result)


# In[86]:


Result = Result.reset_index()


# In[87]:


col_names = ['Model','Accuracy']
Result.columns = col_names


# In[88]:


Result


# #### Converting the processed dataset df for further exploration purpose

# In[89]:


#df.to_csv('Health_care.csv')


# In[90]:


pip install lazypredict


# In[91]:


from lazypredict.Supervised import LazyClassifier
clf = LazyClassifier(verbose=0,ignore_warnings=True,custom_metric=None)


# In[93]:


model,predictions =clf.fit(x_train,x_test,y_train,y_test)


# In[94]:


print(model)


# In[ ]:





# In[ ]:





# In[ ]:




