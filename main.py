import math
import numpy as np
import pandas as pd
import seaborn as seaborn
from matplotlib import pyplot as matplot
from sklearn.metrics import *
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import cross_val_score, cross_val_predict

"""
Group members:

Andy Nguyen
Rocio Salguero
Annie Chen

Class:

CPSC 483-01

Sources:

http://www.ritchieng.com/machine-learning-evaluate-linear-regression-model/
http://scikit-learn.org/stable/modules/generated/sklearn.model_selection.cross_val_score.html
https://stackoverflow.com/questions/22239691/code-for-line-of-best-fit-of-a-scatter-plot-in-python
"""

# Removes scientific notation
np.set_printoptions(suppress=True)

# Loading
data = pd.read_csv("Dataset.csv")
x_title = ['Tm', ' Pr', 'Th', 'Sv']
y_title = 'Idx'
x_original = data[x_title]
y_original = data[y_title]

# Save the values for plotting
mse_Plot = []
mae_Plot = []
rmse_Plot = []
variance_Plot = []

"""
Linear & Non-linear model (Using least squares)
"""
for order in range(1, 7):
    # Change order of polynomial (Order 1 is linear)
    poly = PolynomialFeatures(order)
    x_poly = poly.fit_transform(x_original)

    # Least Square Model (LS Model)
    LSModel = LinearRegression()
    LSModel.fit(x_poly, y_original)
    cv_score = cross_val_score(LSModel, x_poly, y_original, cv=10)
    # Original predict uses linear model
    # y_predict = LSModel.predict(x_original)
    # Uses cross validation estimations
    y_predict = cross_val_predict(LSModel, x_poly, y_original, cv=10)
    variance = y_predict.std() ** 2

    # Measure errors
    MSE = mean_squared_error(y_original, y_predict)
    MAE = mean_absolute_error(y_original, y_predict)
    RMSE = math.sqrt(MSE)
    R_sq = r2_score(y_original, y_predict)
    cv_accuracy = (cv_score.mean(), cv_score.std() * 2)

    # Add to graph
    mse_Plot.append(MSE)
    rmse_Plot.append(RMSE)
    mae_Plot.append(MAE)
    variance_Plot.append(variance)

    """
    Output

    # Note: Can be verbose, uncomment as needed
    """

    # Least squares output
    print("Polynomial order of", str(order) + ":")
    # print("Y Predictions: ", y_predict)
    # print("Intercept: ", LSModel.intercept_)
    # print("Coef: ", LSModel.coef_)
    # print("Standard Deviation = ", y_predict.std())
    print("Variance = ", variance)

    # Error estimations
    print("Mean Absolute Error (MAE) = ", MAE)
    print("Mean Squared Error (MSE) = ", MSE)
    print("Root Mean Squared Error (RMSE) = ", RMSE)
    # print("Explained Variance Score = ", explained_variance_score(y_original, y_predict))
    print("R-squared = ", R_sq)
    print("Accuracy & 95% CI: " + "%0.3f (+/- %0.3f)" % cv_accuracy, "\n")

    """
    Graph: Best fit line for actual and LSM predicted values

    # Note: outputs 6 graphs, uncomment
    """
    matplot.scatter(y_original, y_predict, color='black')
    matplot.plot(np.unique(y_original), np.poly1d(np.polyfit(y_original, y_original, 1))(np.unique(y_original)),
                 color='blue',
                 linewidth=3)
    matplot.title("Best fit with polynomial order " + str(order))
    matplot.xlabel("Actual values")
    matplot.ylabel("Predicted values")
    matplot.show()

"""
Other graphs:

1. Features from data set
2. Best fit line of features
3. Line graph of error estimates & order
4. Line graph variance & order
"""

# 1. Features from data set

# seaborn.pairplot(data, x_vars=x_title, y_vars=y_title, size=7, aspect=1)
# matplot.show()

# 2. Best fit line of features

# x_labels = ['Temperature', 'Pressure', 'Thermal Conductivity', 'Sound Velocity']
# for count, x_attr in enumerate(x_title):
#     matplot.scatter(x_original[x_attr], y_original)
#     liLSM = LinearRegression()
#     liLSM.fit(x_train[x_attr].values.reshape(-1, 1), y_train)
#     y_predict = liLSM.predict(x_test[x_attr].values.reshape(-1, 1))
#     matplot.plot(x_test[x_attr].values.reshape(-1, 1), y_predict, 'r')
#     matplot.legend(['Predicted line', 'Observed data'])
#     matplot.xlabel(x_labels[count])
#     matplot.ylabel('Chem Index')
#     matplot.show()

# 3. Line graph of error estimates & order

order = [1, 2, 3, 4, 5, 6]
colors = ['blue', 'black', 'red', 'green']
eval_x = [mae_Plot, mse_Plot, rmse_Plot]
for count in range(0, 3):
    matplot.plot(order, eval_x[count], linestyle='-', marker='.', color=colors[count])
    matplot.xticks(np.arange(min(order), max(order) + 1, 1.0))
matplot.title("Error estimation as order increases")
matplot.xlabel('Order of polynomial')
matplot.ylabel('Error estimate %')
matplot.legend(['Mean Absolute Error (MAE)', 'Mean Squared Error (MSE)', 'Root Mean Squared Error (RMSE)'])
matplot.show()

# 4. Line graph variance & order
matplot.plot(order, variance_Plot, linestyle='-', marker='.', color='magenta')
matplot.xticks(np.arange(min(order), max(order) + 1, 1.0))
matplot.title("Change in variance")
matplot.xlabel('Order of polynomial')
matplot.ylabel('Change in variance')
matplot.legend(['Variance'])
matplot.show()
