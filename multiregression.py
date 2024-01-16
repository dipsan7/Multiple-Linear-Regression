import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
dataset=pd.read_csv("C:\\Users\Dipsan\\Desktop\\multiple regression\\50_Startups.csv")
X=dataset.iloc[:,:-1].values
y=dataset.iloc[:,-1].values
print(X)
print(y)
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
ct=ColumnTransformer(transformers=[('encoder',OneHotEncoder(),[3])],remainder='passthrough')
X=np.array(ct.fit_transform(X))
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test =train_test_split(X,y,test_size=0.2,random_state=0)
from sklearn.linear_model import LinearRegression
Rregressor = LinearRegression()
Rregressor.fit(X_train,y_train)
y_pred = Rregressor.predict(X_test)
np.set_printoptions(precision=2)  #to print 2 comma value
print(np.concatenate((y_pred.reshape(len(y_pred),1),y_test.reshape(len(y_test),1)),1))