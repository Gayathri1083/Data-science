# -*- coding: utf-8 -*-
"""
Created on Sat Jul 11 17:30:10 2020

@author: Gayathri
"""
#import the necessary packages
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
from sklearn.impute import KNNImputer

#load the dataset
df = pd.read_csv("C:\\Gayathri\\PHD\diabetes_data.csv",header=0)
print(df.head())
print(df.columns)

#drop the first column as it is unnecessary
df = df.drop('Unnamed: 0',axis=1)
print(df.columns)
print(df.shape)

#check for missing data percentage
print(df.isnull())
missingratio = defaultdict()
for i in df:
    ratio = df[i].isnull().sum()/df.shape[0]
    missingratio[i] = ratio
print(missingratio)

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
#fill in the missing values of frame column with mode as it is categorical
df['frame'][df['frame'].isnull()] = 'medium'
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
#check if all the values are imputed
print(df.isnull().sum())
#remove the observations without classification labels as they are not required for exploratory data analysis
x = df['dm'].isnull()
df = df[~x]

#get the summary statistics
decstat = df.describe().T
#check for the differences in mean and median
decstat[['mean','50%']]

#understand the variables distribution
#The column 'time.ppn' has huge difference between its mean and median. So check it
sns.boxplot(data = df['time.ppn'])
sns.distplot(df['time.ppn'])

#other variables
sns.boxplot(data = df['chol'])
sns.distplot(df['chol'])

sns.boxplot(data = df['stab.glu'])
sns.distplot(df['stab.glu'])

sns.boxplot(data = df['hdl'])
sns.distplot(df['hdl'])

sns.boxplot(data = df['ratio'])
sns.distplot(df['ratio'])

sns.boxplot(data = df['glyhb'])
sns.distplot(df['glyhb'])

sns.boxplot(data = df['age'])
sns.distplot(df['age'])

sns.boxplot(data = df['height'])
sns.distplot(df['height'])

sns.boxplot(data = df['weight'])
sns.distplot(df['weight'])

sns.boxplot(data = df['bp.1s'])
sns.distplot(df['bp.1s'])

sns.boxplot(data = df['bp.1d'])
sns.distplot(df['bp.1d'])

sns.boxplot(data = df['waist'])
sns.distplot(df['waist'])

sns.boxplot(data = df['hip'])
sns.distplot(df['hip'])

#The values form a linear line for these 2 features.
sns.boxplot(data = df['bp.2s'])
sns.boxplot(data = df['bp.2d'])

#check for unique values
df['location'].unique()
df['insurance'].unique()
df['fh'].unique()
df['smoking'].unique()

#check for the relationships between numerical features
#remove the id
dfminusid = df.iloc[:,1:]
#remove the categorical features
dfremfeat = dfminusid[['frame','insurance','fh','smoking']]
x = dfminusid.columns.isin(list(dfremfeat))
dfsub = dfminusid.iloc[:,~x]
sns.pairplot(dfsub)

#check for correlated features
dfcor = dfminusid.corr()
sns.heatmap(dfcor)

#Get the upper triangle as the matrix is symmetrical
upper = dfcor.where(np.triu(np.ones(dfcor.shape), k=1).astype(np.bool))

#get highly correlated
highlycor = [column for column in upper.columns if any(upper[column]>0.7)]
print(highlycor)

#data preprocessing
#convert the object to categorical columns 
df['location']=pd.Categorical(df['location'])
df['frame']=pd.Categorical(df['frame'])
df['dm']=pd.Categorical(df['dm'])
df['gender']=pd.Categorical(df['gender'])

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

# prepare the predictor and response variables
from sklearn import preprocessing
X = df.drop(['id','dm'],axis=1)
Y = df['dm']

#standardise the predictor variables
scalar = preprocessing.StandardScaler().fit(X)
X_scaled=scalar.transform(X) 

#train test data split
from sklearn.model_selection import train_test_split

# Get the 1-dimensional flattened array of response feature
Y = Y.ravel()

X_train, X_test, y_train,y_test = train_test_split(X_scaled,Y, test_size=0.25, random_state=1)

#fit logistic regression model
from sklearn import linear_model
from sklearn.metrics import confusion_matrix,accuracy_score
lm = linear_model.LogisticRegression()
model = lm.fit(X_train,y_train)
y_pred = model.predict(X_test)
print(confusion_matrix(y_test,y_pred))
print(accuracy_score(y_test,y_pred))










