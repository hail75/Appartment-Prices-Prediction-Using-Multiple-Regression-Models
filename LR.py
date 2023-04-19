import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import *
from sklearn.linear_model import *

data = pd.read_csv('dataset.csv', index_col=0)


x = data.drop('price', axis = 1)
y = data['price']
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 28)



#RFR_result = RFR(x_train, x_test, y_train, y_test, 100)

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


result = []
for i in range (21):
    result.append([i, L2LR(x_train, x_test, y_train, y_test, i), L1LR(x_train, x_test, y_train, y_test, i)])

constant = [row[0] for row in result]
l2lr = [row[1] for row in result]
l1lr = [row[2] for row in result]

plt.plot(constant, l2lr, label='Ridge')
plt.plot(constant, l1lr, label='Lasso')
plt.legend()
plt.xlabel("Lambda")
plt.ylabel("Mean Absolute Percentage Error (%)")
plt.show()

