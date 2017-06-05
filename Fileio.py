#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun May 14 16:05:38 2017

@author: siddharth
"""

import pandas as pd
import warnings
warnings.filterwarnings('ignore')

from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.naive_bayes import GaussianNB
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier


loc=r"/home/siddharth/Downloads/train.csv"
loc1=r"/home/siddharth/Downloads/test.csv"
train=pd.read_csv(loc)
test1=pd.read_csv(loc1)


def get_combined_data():
    train=pd.read_csv(loc)
    test=pd.read_csv(loc1)

    target=train.Survived
    train.drop(['Survived'],1 ,inplace=True)


    combined=train.append(test)
    combined.reset_index(inplace=True)
    combined.drop('index',inplace=True,axis=1)
    return combined



combined=get_combined_data()


def drop_columns():
    combined.drop(['Names','Ticket','Cabin','PassengerId','Embarked'],1,inplace=True)
    return combined

drop_columns()

def get_sex():
    combined['Sex']=combined['Sex'].map({'male':0,'female':1})
    return combined
get_sex()


def get_ages():
    combined['Age'].fillna(combined['Age'].median(), inplace=True)
    return combined
get_ages()

def get_fare():
    combined['Fare'].fillna(combined['Fare'].median(), inplace=True)
get_fare()






print combined.shape

print combined.describe()


def recover_train_test_target():
    global combined
    
    train0 = pd.read_csv(loc)
    
    targets = train0.Survived
    train = combined.ix[0:890]
    test = combined.ix[891:]
    
    return train,test,targets

train,test,targets = recover_train_test_target()



#clf=GaussianNB()
#clf=svm.SVC(C=10000.0,kernel='rbf')
clf = RandomForestClassifier(n_estimators=10)


clf = clf.fit(train, targets)
pred=clf.predict(test)


df_output=pd.DataFrame()
df_output['PassengerId']=test1['PassengerId']
df_output['Survived']=pred


df_output[['PassengerId','Survived']].to_csv("/home/siddharth/Downloads/result_Titanivc.csv",index=False)


