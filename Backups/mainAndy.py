import pandas as pd
import numpy as np
import math
from matplotlib import pyplot as matplot
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import *
from sklearn.model_selection import train_test_split, cross_val_score

# Removes scientific notation
np.set_printoptions(suppress=True)

# Loading data
data = pd.read_csv("Dataset.csv")
x_title = ['Tm', ' Pr', 'Th', 'Sv']
y_title = 'Idx'
x_original = data[x_title]
y_original = data[y_title]
x_train, x_test, y_train, y_test = train_test_split(x_original, y_original, test_size=0.5, random_state=1)

# Change order of polynomial (Default is 1)
order = 1
poly = PolynomialFeatures(order)
x_train = poly.fit_transform(x_train)
x_test = poly.fit_transform(x_test)

# Linear Least Square Model (LSM)
linearLSM = LinearRegression()
linearLSM.fit(x_train, y_train)
cv_score = cross_val_score(linearLSM, x_original, y_original, cv=10)
y_predict = linearLSM.predict(x_test)

# Measure errors
MAE = mean_absolute_error(y_test, y_predict)
MSE = mean_squared_error(y_test, y_predict)
RMSE = math.sqrt(MSE)

"""
Output
"""

# LSM output
# print("Y Predictions: ", y_predict)
print("Intercept: ", linearLSM.intercept_)
print("Coef: ", linearLSM.coef_)
# print("Standard Deviation = ", y_test.std())
print("Variance = ", y_test.std() ** 2)

# Error estimations
print("Mean Absolute Error (MAE) = ", MAE)
print("Mean Squared Error (MSE) = ", MSE)
print("Root Mean Squared Error (RMSE) = ", RMSE)
# print("Explained Variance Score = ", explained_variance_score(y_test, y_predict))
print("R-squared  = ", r2_score(y_test, y_predict))
print("Cross-Val Accuracy: %0.2f (+/- %0.2f)" % (cv_score.mean(), cv_score.std() * 2))

# Plotting best fit line
matplot.scatter(y_test, y_predict, color='black')
matplot.plot(np.unique(y_test), np.poly1d(np.polyfit(y_test, y_test, 1))(np.unique(y_test)), color='blue', linewidth=3)
matplot.title("Best fit with degree of " + str(order))
matplot.xlabel("Actual values")
matplot.ylabel("Predicted values")
matplot.show()
