# -*- coding: utf-8 -*-
"""
Created on Tue Oct 30 12:59:21 2018

@author: arvgoyal
Predict the class of the flower based on available attributes
iris.data
"""

import numpy as np
import pandas as pd
import os

os.chdir ("D:\Arvin\ArtInt\input files\AV")

iris = pd.read_csv('iris.csv', header = 'infer' )
del iris
iris = pd.read_csv('iris.csv', header = None )
iris = pd.read_csv('iris.csv', header = [1,2] )

iris.rename(columns={0: 'Var1', 1:'var2', 2:'var3', 3:'var4', 4:'Var5' }, inplace=True)
iris.rename(columns={'Var1': 'var1', 'Var5': 'flwr' }, inplace=True)
iris = pd.read_csv('iris.csv', header = None, names = ['sepal_len','sepal_wdth','petal_len','petal_wdth','flwr'] )



"""
from sklearn.datasets import load_iris
iris1 = load_iris()
# Let's convert to dataframe
df = pd.DataFrame(data= np.c_[iris['data'], iris['target']],
columns= iris['feature_names'] + ['species'])
"""



iris.info()
iris.describe()
iris.head(10) 

iris.iloc[:,4].unique()
iris.loc[:,'flwr'].unique()
iris.flwr.unique()


## Converting flower name to numeric value
from sklearn import preprocessing
le = preprocessing.LabelEncoder()
iris['flwr']= le.fit_transform(iris['flwr'])

## Checking null values
iris.iloc[0].isnull().sum()
iris.iloc[1].isnull().sum()
iris.iloc[2].isnull().sum()
iris.iloc[3].isnull().sum()
iris.iloc[4].isnull().sum()


## Cheking distribution
import matplotlib.pyplot as plt
import seaborn as sns     # seaborn and booke are also visulization packages like matplotlib
#% matplotlib inline       # to get the plot in same page otherwise it will popup

#Plot the probability density of variables
sns.distplot(iris['sepal_len'], hist=True, kde=True, kde_kws = {'linewidth':3})

plt.hist(iris.sepal_len,bins=25)
plt.hist(iris.sepal_wdth,bins=25)
plt.hist(iris.petal_len,bins=25)
plt.hist(iris.petal_wdth,bins=25)
plt.hist(iris.flwr,bins=25)

# transform the variable 3 and 4 as they are not in normal distribution
from scipy.stats import boxcox
iris.petal_len_a = boxcox(iris.petal_len, 1)
plt.hist(iris.petal_len_a,bins=25)
del iris.petal_len_a




##  checking if any of the independent varaiables are correlated.
import pylab as plt
iris.corr()
plt.matshow(iris.corr())

plt.scatter(iris['petal_len'], iris['petal_wdth'])
plt.scatter(iris['sepal_len'], iris['petal_len'])
plt.scatter(iris['sepal_len'], iris['petal_wdth'])
plt.show()


## Checking any outlier
plt.boxplot(iris.iloc[0], sym='gx', notch=False)
plt.boxplot(iris.iloc[1], sym='gx', notch=False)
plt.boxplot(iris.iloc[2], sym='gx', notch=False)
plt.boxplot(iris.iloc[3], sym='gx', notch=False)


# var3 and var4 are related, var4 is more closely realted to output varaible hence will drop var3
# var3 and var1 related, however we are already deleting var3
# var1 and var4 are also related, however we are having very less varaible. going to keep both for now.

iris.columns
del iris.petal_len
del iris['petal_len']
del iris['sepal_wdth']




#Test Train Split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(iris.iloc[:,:3], iris.iloc[:,3], test_size = 0.25, random_state = 0)
X_train, X_test, y_train, y_test = train_test_split(iris.iloc[:,:4], iris.iloc[:,4], test_size = 0.25, random_state = 0)

# Create a random forest Classifier.
from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier(n_jobs=2, random_state=0)
clf.fit(X_train, y_train)


list (zip(X_train, clf.feature_importances_))

#prodiction from model
preds = clf.predict(X_test)

# this will gaive the probability for all three classes in target variable based X_test
clf.predict_proba(X_test)[0:10]

# some short of confusion matrics
pd.crosstab(y_test, preds, rownames=['Actual Species'], colnames=['Predicted Species'])





