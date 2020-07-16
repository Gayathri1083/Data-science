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

#check for the datatypes of each column is they are correct
print(df.dtypes)
#convert frame column to category
df['frame'] = pd.Categorical(df['frame'])
#Now,convert it to numeric
df['frame'] = df['frame'].cat.codes

#convert dm column to category
df['dm'] = pd.Categorical(df['dm'])
#Now,convert it to numeric
df['dm'] = df['dm'].cat.codes


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
#fill in the missing values of frame column with median
df['frame'].fillna(df['frame'].median(),inplace=True)
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

sns.boxplot(data = df['frame'])
sns.distplot(df['frame'])

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

highlycor = [column for column in upper.columns if any(upper[column]>0.7)]
print(highlycor)







