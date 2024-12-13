# -*- coding: utf-8 -*-
"""
Created on Thu Dec 12 18:31:28 2024

@author: DELL
"""

import pandas as pd
import numpy as np
from keras import Sequential, Input
from sklearn import metrics
from scikeras.wrappers import KerasRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam, RMSprop

######################################数据预处理############################################
train_data = pd.read_csv("D:/myfiles/yanyi_kaggle/cug-24-fallasm/train.csv")
test_data = pd.read_csv("D:/myfiles/yanyi_kaggle/cug-24-fallasm/test.csv")
train_data = train_data.fillna(train_data.mean())
train_data_array = np.array(train_data)
test_data = test_data.fillna(test_data.mean())
test_data_array = np.array(test_data)

#训练集测试集
b = train_data_array[:, 21]
a = train_data_array[:, 0:21]
x_train, x_test, y_train, y_test = train_test_split(a, b, test_size = 0.3, random_state = 1)
#数据标准化
scaler = StandardScaler()
X_train = scaler.fit_transform(x_train)
X_test = scaler.transform(x_test)

###########################采用网格搜索和交叉验证来确定最优参数###############################
## 定义构建模型的函数
#def create_model(learning_rate = 0.001):
#    model = Sequential()
#    model.add(Input(shape = (X_train.shape[1],)))  # 明确定义输入形状
#    model.add(Dense(units = 64, activation = 'relu'))  # 输入层和第一个隐藏层
#    model.add(Dense(units = 32, activation = 'relu'))  # 第二个隐藏层
#    model.add(Dense(units = 1, activation = 'linear'))  # 输出层
#    model.compile(loss = 'mean_squared_error', optimizer = Adam(learning_rate=learning_rate))
#    return model

## 包装模型
#model = KerasRegressor(model = create_model, epochs = 100, batch_size = 10, verbose = 0)

## 定义参数网络
#param_grid = {
#    'model__learning_rate': [0.001, 0.01, 0.1],
#    'epochs': [50, 100],
#    'batch_size': [10, 20]
#}

## 创建GridSearchCV对象
#grid = GridSearchCV(estimator = model, param_grid = param_grid, cv = 3, scoring = 'neg_mean_squared_error')

## 执行网格搜索
#grid_result = grid.fit(X_train, y_train)

## 输出最优参数
#print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))

#################################使用最优参数构建模型########################################
#best_model = grid_result.best_estimator_

#predict = best_model.predict(X_test) #预测
#predict = predict.flatten() #使predict一维
#r2 = metrics.r2_score(y_test, predict)
#print(r2)

#参数选择耗时较长

#搭建BP神经网络
model = Sequential()
model.add(Dense(units=64, activation='relu', input_shape=(X_train.shape[1],))) # 输入层和第一个隐藏层
model.add(Dense(units=32, activation='relu')) # 第二个隐藏层
model.add(Dense(units=1, activation='linear')) # 输出层

#编译模型
model.compile(loss='mean_squared_error', optimizer='adam') # 使用均方误差作为损失函数，Adam优化器

#训练模型
history = model.fit(X_train, y_train, epochs=20, batch_size=10, validation_split=0.1)

predictions = model.predict(X_test)
predictions = predictions.flatten()
r2 = metrics.r2_score(y_test, predictions)
print(r2)