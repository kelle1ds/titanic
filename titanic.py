# -*- coding: utf-8 -*-

#import numpy as np
import pandas as pd
import os
import ujson
import falcon

class GetInfo:
    def on_post(self, req,resp):
        ext = req.content_type[6:]
        

#from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV

train = pd.read_csv("train_titanic.csv")  #training data
person = [[900, 3, 'Kelley Dixie', 'Female', 51.5, 0, 1, 33098, 12.0,'','S']]
test = pd.DataFrame(person, columns=['PassengerId','Pclass','Name','Sex','Age','SibSp','Parch','Ticket','Fare','Cabin','Embarked'])
train.drop(['Name','Ticket','Cabin','Embarked'],axis='columns', inplace=True)
test.drop(['Name','Ticket','Cabin','Embarked'],axis='columns', inplace=True)

mean_value = train['Age'].mean()
train['Age'] = train['Age'].fillna(mean_value)
test['Age'] = test['Age'].fillna(mean_value)
mean_value_fare = test['Fare'].mean()
train['Fare'] = train['Fare'].fillna(mean_value_fare)
test['Fare'] = test['Fare'].fillna(mean_value_fare)

train = pd.get_dummies(train, columns = ['Sex'])
test = pd.get_dummies(test, columns = ['Sex'])

train = train.drop(['Sex_male'], axis=1)
train = train.rename(columns={"Sex_female": "Sex"})
columns = test.columns.values.tolist()

if columns[6] == 'Sex_Male':
    test = test.rename(columns={"Sex_Male": "Sex"})
if columns[6] == 'Sex_Female':
    test = test.rename(columns={"Sex_Female": "Sex"})


#get passenger id for each set then drop feature from sets
test_id = test.iloc[:,0]
train_id = train.iloc[:,0]

#drop passenger id for each set

test.drop('PassengerId',axis='columns', inplace=True)
train.drop('PassengerId',axis='columns', inplace=True)

y = train.iloc[:,0]
X = train.iloc[:,1:13]

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.30, random_state=42)

parameters = {'criterion':['gini','entropy'],'max_depth':[2,3,4], 'min_samples_leaf':[4,5,6,7], 'min_samples_split':[3,4,5,6,7]}

dt = DecisionTreeClassifier()
clf = GridSearchCV(dt, parameters)
clf.fit(X_train,y_train)

y_train_pred = clf.predict(X_train)
y_test_pred = clf.predict(X_test)

# Calculate the accuracy
train_accuracy = accuracy_score(y_train, y_train_pred)
test_accuracy = accuracy_score(y_test, y_test_pred)

#print('The training accuracy is', train_accuracy)
#print('The test accuracy is', test_accuracy)

result = clf.predict(test)
print(result)