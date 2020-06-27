# -*- coding: utf-8 -*-
"""
Created on Sat Jun 27 17:30:16 2020

@author: AATL
"""

#IRIS DATASET


# importação
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score


# carregamento do dataset
data = pd.read_csv('Iris.csv')
print(data.head())

print('\n\nColumn Names\n\n')
print(data.columns)

#atribuição das variaveis
encode = LabelEncoder()
data.species = encode.fit_transform(data.species)

print(data.head())

# train-test-split   
train , test = train_test_split(data,test_size=0.2,random_state=0)

print('shape of training data : ',train.shape)
print('shape of testing data',test.shape)

# separação
train_x = train.drop(columns=['species'],axis=1)
train_y = train['species']

test_x = test.drop(columns=['species'],axis=1)
test_y = test['species']

# criação do objeto para o modelo de ML
model = LogisticRegression()

model.fit(train_x,train_y)

predict = model.predict(test_x)

print('Predicted Values on Test Data',encode.inverse_transform(predict))

print('\n\nAccuracy Score on test data : \n\n')
print(accuracy_score(test_y,predict))
