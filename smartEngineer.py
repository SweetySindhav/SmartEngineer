# -*- coding: utf-8 -*-
"""
Created on Wed Apr 17 08:16:54 2019

@author: sweetysindhav
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
#from sklearn.metrics import accuracy

pd.set_option('precision',3)
pd.set_option('display.width',100)

#Load the data
df_train = pd.read_csv("train_data.csv")
df_test = pd.read_csv("test_data.csv")

#Analyzing the data
print(df_train.describe())
print(df_train.columns)
print(df_train.info())

#Find the correlation among features

#x = [c for c in df_train.columns if c is not 'E']
x = df_train[['T','V','P','RH']]
y = df_train['E']

#for col in df_train:
#    plt.scatter(col,y,style='o',color = 'Blue')
#    plt.title("{} vs E".format(col))
#    plt.xlabel(col)
#    plt.ylabel('E')
#    plt.show()

#print("The feature selected are: {} \n The target variable is:{}".format(x,y))


corr = df_train.corr()
print(corr)
figure = plt.figure(figsize=(8,8))
sns.heatmap(corr,annot=True)
plt.title("Correlation Matrix")
plt.show()

#Training the algorithm
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.3,random_state=10)
clf = LinearRegression().fit(x_train,y_train)
prediction = clf.predict(x_test)

#print("Prediction are :\n{}".format(prediction.sample(5)))

#Evaluating our model
Accuracy = clf.score(x_test,y_test)
print("Accuracy is :{}".format(Accuracy))

predict_test = clf.predict(df_test)

#df_submission = pd.read_csv("test_prediction.csv")
df_submission = pd.DataFrame(predict_test)
df_submission.to_csv("test_prediction.csv",header = False,index=False)
print("File test_prediction created...")
