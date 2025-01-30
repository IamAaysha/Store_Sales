# -*- coding: utf-8 -*-
"""
Created on Thu May  7 04:19:54 2020

@author: DELL
"""


import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error as mse
train_data = pd.read_csv('E:\\spider\\dataset\\train.csv')
test_data = pd.read_csv('E:\\spider\\dataset\\test.csv')
train_data.isna().sum()
test_data.isna().sum()
train=train_data.dropna()
test=test_data.dropna()
print('\n Shape of training data=',train.shape)#dimentions
print('\n Shape of training data=',test.shape)
train_x=train.drop(columns=['Item_Outlet_Sales'],axis=1)
train_y=train['Item_Outlet_Sales']
test_x=test.drop(columns=['Item_Outlet_Sales'],axis=1)
test_y=test['Item_Outlet_Sales']
model=LinearRegression()
model.fit(train_x,train_y)
print('\n Cofficient of Model:'model.coef_)
predict_train=model.predict(train_x)
predict_test=model.predict(test_y)
print('\n Item_Outlet_Sales on training data=',predict_train)
print('\n Item_Outlet_Sales on test data=',predict_test)
rmse_train = mean_squared_error(train_y,predict_train)**(0.5)
print('\n RMSE on train dataset : ', rmse_train)
rmse_test = mean_squared_error(test_y,predict_test)**(0.5)
print('\n RMSE on train dataset : ', rmse_test)
###########
from sklearn.preprocessing import LabelEncoder
lm=LabelEncoder()
cate=['Item_Identifier','Item_Weight','Item_Fat_Content','Item_Type','Outlet_Identifier','Outlet_Size','Outlet_Location_Type','Outlet_Type']
for i in cate:
    train[i]=lm.fit_transform(train[i])
   
from sklearn.preprocessing import MinMaxScaler
mm=MinMaxScaler()
conti=['Item_Visibility','Item_MRP','Outlet_Establishment_Year','Item_Outlet_Sales']
for i in conti:
    train[i]=mm.fit_transform(train[i].values.reshape(-1,1))

