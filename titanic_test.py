# -*- coding: utf-8 -*-

 

import pandas as pd

from joblib import load
import sklearn

 
person = [[900, 3, 'Kelley Dixie', 'Female', 51.5, 0, 1, 33098, 12.0,'','S']]

test = pd.DataFrame(person, columns=['PassengerId','Pclass','Name','Sex','Age','SibSp','Parch','Ticket','Fare','Cabin','Embarked'])

test.drop(['Name','Ticket','Cabin','Embarked'],axis='columns', inplace=True)


test = pd.get_dummies(test, columns = ['Sex'])

columns = test.columns.values.tolist()

if columns[6] == 'Sex_Male':

    test = test.rename(columns={"Sex_Male": "Sex"})

if columns[6] == 'Sex_Female':

    test = test.rename(columns={"Sex_Female": "Sex"})

 
test.drop('PassengerId',axis='columns', inplace=True)

clf = load('filename.joblib')

result = clf.predict(test)
print(sklearn.__version__)