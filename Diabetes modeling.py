#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
from sklearn.impute import KNNImputer


# In[8]:


df = pd.read_csv("C:\\Gayathri\\PHD\diabetes_data.csv",header=0)


# In[9]:


print(df.head())


# In[10]:


print(df.columns)


# In[11]:


#drop the first column as it is unnecessary
df = df.drop('Unnamed: 0',axis=1)


# In[12]:


print(df.columns)


# In[13]:


print(df.shape)


# In[14]:


print(df.info())


# In[15]:


print(df.describe())


# In[16]:


#check for missing data percentage
print(df.isnull())


# In[17]:


missingratio = defaultdict()
for i in df:
    ratio = df[i].isnull().sum()/df.shape[0]
    missingratio[i] = ratio
print(missingratio)


# In[18]:


#impute missing data columns with mean values - for columns where missingratio is less
df['chol'].fillna(df['chol'].mean(),inplace=True)
df['hdl'].fillna(df['hdl'].mean(),inplace=True)
df['ratio'].fillna(df['ratio'].mean(),inplace=True)
df['glyhb'].fillna(df['glyhb'].mean(),inplace=True)
df['height'].fillna(df['height'].mean(),inplace=True)
df['weight'].fillna(df['weight'].mean(),inplace=True)
df['bp.1s'].fillna(df['bp.1s'].mean(),inplace=True)
df['bp.1d'].fillna(df['bp.1d'].mean(),inplace=True)
df['waist'].fillna(df['waist'].mean(),inplace=True)
df['hip'].fillna(df['hip'].mean(),inplace=True)
df['time.ppn'].fillna(df['time.ppn'].mean(),inplace=True)


# In[19]:


df.info()


# In[20]:


#fill in the missing values of frame column with mode as it is categorical
df['frame'][df['frame'].isnull()] = 'medium'


# In[21]:


df.info()


# In[22]:


#impute the other 2 columns with knn imputation as the missing ratio percentage is more
#convert them into numpy arrays
x = df['bp.2d'].to_numpy()
x = x.reshape(-1,1)
imputer = KNNImputer(n_neighbors=5)
x = imputer.fit_transform(x)
df['bp.2d'] = x
x = df['bp.2s'].to_numpy()
x = x.reshape(-1,1)
imputer = KNNImputer(n_neighbors=5)
x = imputer.fit_transform(x)
df['bp.2s'] = x


# In[23]:


df.info()


# In[24]:


#check if all the values are imputed
print(df.isnull().sum())


# In[25]:


#remove the observations without classification labels as they are not required for exploratory data analysis
x = df['dm'].isnull()
df = df[~x]


# In[26]:


df.info()


# In[27]:


#Algorithm likes numbers
df['dm'] = df['dm'].map(dict(yes=1, no=0))


# In[28]:


df.info()


# In[29]:


#get the summary statistics
decstat = df.describe().T
print(decstat)


# In[30]:


#check for the differences in mean and median
decstat[['mean','50%']]


# In[31]:


#understand the variables distribution
#The column 'time.ppn' has huge difference between its mean and median. So check it
sns.boxplot(data = df['time.ppn'])


# In[32]:


#other variables
sns.boxplot(data = df['chol'])


# In[33]:


sns.boxplot(data = df['stab.glu'])


# In[34]:


sns.boxplot(data = df['hdl'])


# In[35]:


sns.boxplot(data = df['ratio'])


# In[36]:


sns.boxplot(data = df['glyhb'])


# In[37]:


sns.boxplot(data = df['age'])


# In[39]:


sns.boxplot(data = df['height'])


# In[40]:


sns.boxplot(data = df['weight'])


# In[41]:


sns.boxplot(data = df['bp.1s'])


# In[38]:


sns.boxplot(data = df['bp.1d'])


# In[39]:


sns.boxplot(data = df['waist'])


# In[40]:


sns.boxplot(data = df['hip'])


# In[41]:


#The values form a linear line for these 2 features.
sns.boxplot(data = df['bp.2s'])


# In[42]:


#The values form a linear line for these 2 features.
sns.boxplot(data = df['bp.2d'])


# In[42]:


sns.pairplot(df)


# In[43]:


plt.figure(figsize=(10,6))
p = sns.countplot(x='age',data=df)
p.set_xticklabels(p.get_xticklabels(),rotation=75)


# In[44]:


p = sns.countplot(x='age',hue='dm',data=df)
p.set_xticklabels(p.get_xticklabels(),rotation=75)
for ind, label in enumerate(p.get_xticklabels()):
    if ind % 10 == 0:  # every 10th label is kept
        label.set_visible(True)
    else:
        label.set_visible(False)


# In[45]:


plt.figure(figsize=(10,8))
p = sns.boxplot(x='hdl',y='age',hue = 'dm', data=df)
p.set_xticklabels(p.get_xticklabels(),rotation=75)
for ind, label in enumerate(p.get_xticklabels()):
    if ind % 10 == 0:  # every 10th label is kept
        label.set_visible(True)
    else:
        label.set_visible(False)


# In[46]:


sns.jointplot(x = 'time.ppn',y='age',data=df)


# In[71]:



#check for unique values
df['location'].unique()


# In[72]:


df['insurance'].unique()


# In[73]:


df['fh'].unique()


# In[74]:


df['smoking'].unique()


# In[48]:


#check for the relationships between numerical features
#remove the id
dfminusid = df.iloc[:,1:]


# In[49]:


dfminusid.shape


# In[50]:


dfminusid.columns


# In[51]:


#remove the categorical features
dfremfeat = dfminusid[['frame','insurance','fh','smoking']]
x = dfminusid.columns.isin(list(dfremfeat))
dfsub = dfminusid.iloc[:,~x]


# In[52]:


dfsub.shape


# In[53]:


dfsub.columns


# In[81]:


sns.pairplot(dfsub)


# In[54]:


#check for correlated features
dfcor = dfminusid.corr()
sns.heatmap(dfcor)


# In[55]:


#Get the upper triangle as the matrix is symmetrical
upper = dfcor.where(np.triu(np.ones(dfcor.shape), k=1).astype(np.bool))


# In[56]:


#get highly correlated
highlycor = [column for column in upper.columns if any(upper[column]>0.9)]
print(highlycor)


# In[91]:


print(upper)


# In[57]:


df.info()


# In[58]:


#data preprocessing
#convert the object to categorical columns 
df['location']=pd.Categorical(df['location'])
df['frame']=pd.Categorical(df['frame'])
df['gender']=pd.Categorical(df['gender'])


# In[59]:


df.info()


# In[60]:


#get the dummies for categorical variables
dfc = pd.get_dummies(df['location'])
df = pd.concat([df,dfc],axis=1)
df = df.drop('location',axis=1)

dfcf = pd.get_dummies(df['frame'])
df = pd.concat([df,dfcf],axis=1)
df = df.drop('frame',axis=1)

dfcd = pd.get_dummies(df['gender'])
df = pd.concat([df,dfcd],axis=1)
df = df.drop('gender',axis=1)


# In[61]:


df.shape


# In[62]:


df.head()


# In[63]:


# prepare the predictor and response variables
from sklearn import preprocessing
X = df.drop(['id','dm'],axis=1)
Y = df['dm']


# In[100]:


X.head()


# In[101]:


Y.head()


# In[102]:


#standardise the predictor variables
scalar = preprocessing.StandardScaler().fit(X)
X_scaled=scalar.transform(X) 


# In[104]:


#train test data split
from sklearn.model_selection import train_test_split

# Get the 1-dimensional flattened array of response feature
Y = Y.ravel()

X_train, X_test, y_train,y_test = train_test_split(X_scaled,Y, test_size=0.25, random_state=1)


# In[105]:


#fit logistic regression model
from sklearn import linear_model
from sklearn.metrics import confusion_matrix,accuracy_score,classification_report

lm = linear_model.LogisticRegression()
model = lm.fit(X_train,y_train)
y_pred = model.predict(X_test)
print(confusion_matrix(y_test,y_pred))
print(accuracy_score(y_test,y_pred))
print(classification_report(y_test,y_pred))


# In[106]:


#fit naive bayes model
from sklearn.naive_bayes import GaussianNB
NB = GaussianNB()
model = NB.fit(X_train,y_train)
y_pred = model.predict(X_test)
print(confusion_matrix(y_test,y_pred))
print(accuracy_score(y_test,y_pred))
print(classification_report(y_test,y_pred))


# In[107]:


#fit random forest model
from sklearn.ensemble import RandomForestClassifier
RF = RandomForestClassifier()
model = RF.fit(X_train,y_train)
y_pred = model.predict(X_test)
print(confusion_matrix(y_test,y_pred))
print(accuracy_score(y_test,y_pred))
print(classification_report(y_test,y_pred))


# In[108]:


model.feature_importances_


# In[110]:


importance = model.feature_importances_
# summarize feature importance
for i,v in enumerate(importance):
	print('Feature: %0d, Score: %.5f' % (i,v))
# plot feature importance
plt.bar([x for x in range(len(importance))], importance)
plt.show()


# In[ ]:




