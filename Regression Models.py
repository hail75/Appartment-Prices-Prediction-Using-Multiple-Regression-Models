import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import *
from sklearn.linear_model import *
from sklearn.neighbors import KNeighborsRegressor

data = pd.read_csv('dataset.csv', index_col=0)

x = data.drop('price', axis = 1)
y = data['price']
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 28)

def L2LR(x_train, x_test, y_train, y_test, n):
    model = Ridge(alpha = n)
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    y_pred_rounded = [round(result, 2) for result in y_pred]
    mape = float(format(mean_absolute_percentage_error(y_test, y_pred_rounded), '.3f'))
    return mape * 100

def L1LR(x_train, x_test, y_train, y_test, n):
    model = Lasso(alpha = n)
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    y_pred_rounded = [round(result, 2) for result in y_pred]
    mape = float(format(mean_absolute_percentage_error(y_test, y_pred_rounded), '.3f'))
    return mape * 100

def KNRU(x_train, x_test, y_train, y_test, n):
    model = KNeighborsRegressor(n_neighbors= n, weights='uniform')
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    y_pred_rounded = [round(result, 2) for result in y_pred]
    mape = float(format(mean_absolute_percentage_error(y_test, y_pred_rounded), '.3f'))
    return mape * 100

def KNRD(x_train, x_test, y_train, y_test, n):
    model = KNeighborsRegressor(n_neighbors= n, weights='distance')
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    y_pred_rounded = [round(result, 2) for result in y_pred]
    mape = float(format(mean_absolute_percentage_error(y_test, y_pred_rounded), '.3f'))
    return mape * 100

def LR_Graph():
    lr_result = []
    for i in range (21):
        lr_result.append([i, L2LR(x_train, x_test, y_train, y_test, i), L1LR(x_train, x_test, y_train, y_test, i)])

    constant = [row[0] for row in lr_result]
    l2lr = [row[1] for row in lr_result]
    l1lr = [row[2] for row in lr_result]

    plt.plot(constant, l2lr, label='Ridge')
    plt.plot(constant, l1lr, label='Lasso')
    plt.legend()
    plt.xlabel("λ")
    plt.ylabel("Mean Absolute Percentage Error (%)")
    plt.show()

def KNR_Graph():
    knr_result = []
    for i in range (1,51):
        knr_result.append([i, KNRU(x_train, x_test, y_train, y_test, i), KNRD(x_train, x_test, y_train, y_test, i)])

    constant = [row[0] for row in knr_result]
    knru = [row[1] for row in knr_result]
    knrd = [row[2] for row in knr_result]

    plt.plot(constant, knru, label='Uniform weight')
    plt.plot(constant, knrd, label='Distance weight')
    plt.legend()
    plt.xlabel("k")
    plt.ylabel("Mean Absolute Percentage Error (%)")
    plt.show()

A = input("Press a number to see the graph. [1] Linear Regression; [2] K-nearest Neighbors Regression: ")
if A == '1':
    LR_Graph()
if A == '2':
    KNR_Graph()