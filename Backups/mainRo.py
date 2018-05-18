import pandas as pd
import numpy as np
import math
import seaborn as seaborn
from matplotlib import pyplot as matplot
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, explained_variance_score
from sklearn.model_selection import train_test_split, cross_val_score

# Removes scientific notation
np.set_printoptions(suppress=True)

# Loading data
data = pd.read_csv("Dataset.csv")
x_title = ['Tm', ' Pr', 'Th', 'Sv']
y_title = 'Idx'
x_original = data[x_title]
y_original = data.Idx
x_train, x_test, y_train, y_test = train_test_split(x_original, y_original, test_size=0.5, random_state=1)


# Plotting the Data scatter Linear Regression======================================= 
seaborn.pairplot(data, x_vars=x_title, y_vars=y_title, size=7, aspect=1)
matplot.show()

#plotting the original graphs and linear regression
x_attributes = ["Tm", " Pr", "Th", "Sv"]
x_labels = ['Tempurature', 'Pressure', 'Thermal Conductivity', 'Sound Velocity']

for count, x_attr in enumerate(x_attributes):
    matplot.scatter(x_original[x_attr], y_original)
    liLSM = LinearRegression()
    liLSM.fit(x_train[x_attr].reshape(-1,1), y_train)
    y_predict = liLSM.predict(x_test[x_attr].reshape(-1,1))
    matplot.plot(x_test[x_attr].reshape(-1,1), y_predict, 'r') 
    matplot.legend(['Predicted line','Observed data'])
    matplot.xlabel(x_labels[count])
    matplot.ylabel('Chem Index')
    matplot.show()

# Linear Least Square Model (LSM)
linearLSM = LinearRegression()
linearLSM.fit(x_train, y_train)
cv_score = cross_val_score(linearLSM, x_original, y_original, cv=10)
y_predict = linearLSM.predict(x_test)

# Measure errors
MAE = mean_absolute_error(y_test, y_predict)
MSE = mean_squared_error(y_test, y_predict)
RMSE = math.sqrt(MSE)

# Output
# print("Y Predictions: ", y_predict)
print("Linear regression:")
print("Intercept: ", linearLSM.intercept_)
print("Coef: ", linearLSM.coef_)
print("Mean Squared Error = ", MSE)
print("Root Mean Squared Error = ", RMSE)
print("Mean Absolute error = ", MAE)
print("R-squared  = ", r2_score(y_test, y_predict))
# Cross validates the original data set
print("Cross-Val Accuracy: %0.2f (+/- %0.2f)" % (cv_score.mean(), cv_score.std() * 2))
print("Variance = ", y_test.std() ** 2, "/n")

# Plotting best fit line for y values un comment to see order of 1
matplot.scatter(y_test, y_predict, color='black')
matplot.plot(np.unique(y_test), np.poly1d(np.polyfit(y_test, y_test, 1))(np.unique(y_test)), color='blue', linewidth=3)
matplot.title("Best fit Linear order 1")
matplot.xlabel("Actual values")
matplot.ylabel("Predicted values")
matplot.show()

#Nonlinear regression analysis=====================================

# Change order of polynomial 
mse_Plot = []
mae_Plot = []
rmse_Plot = []
variance_Plot = []

for order in range(2, 11):
    poly = PolynomialFeatures(order)
    x_trainnew = poly.fit_transform(x_train)
    x_testnew = poly.fit_transform(x_test)

    # Linear Least Square Model (LSM)
    linearLSM = LinearRegression()
    linearLSM.fit(x_trainnew, y_train)
    cv_score = cross_val_score(linearLSM, x_original, y_original, cv=10)
	y_predict = linearLSM.predict(x_test)

	# Measure errors	
	MSE = mean_squared_error(y_test, y_predict)
	MAE = mean_absolute_error(y_test, y_predict)
	RMSE = math.sqrt(MSE)

    #Save the values for plotting for evaluation 
    mse_Plot.append(mean_squared_error(y_test, y_predict))
    mae_Plot.append(mean_absolute_error(y_test, y_predict))
    rmse_Plot.append(math.sqrt(MSE))
    variance_Plot.append(y_test.std() ** 2)

    # Output
	print("Evaluate on polynomial order: ", order)
    # print("Y Predictions: ", y_predict)
    #print("Intercept: ", linearLSM.intercept_)
    #print("Coef: ", linearLSM.coef_)
    print("Mean Squared Error = ", MSE)
    print("Root Mean Squared Error = ", RMSE)
    print("Mean Absolute error = ", MAE)
    print("R-squared  = ", r2_score(y_test, y_predict))
    # Cross validates the original data set
    print("Cross-Val Accuracy: %0.2f (+/- %0.2f)" % (cv_score.mean(), cv_score.std() * 2))
    print("Variance = ", y_test.std() ** 2, "/n")

    # Plotting best fit line for y values uncomment to see: 
    matplot.scatter(y_test, y_predict, color='black')
    matplot.plot(np.unique(y_test), np.poly1d(np.polyfit(y_test, y_test, 1))(np.unique(y_test)), color='blue', linewidth=3)
    matplot.title("Polynomial order " + str(order))
    matplot.xlabel("Actual values")
    matplot.ylabel("Predicted values")
    matplot.show()

#plotting the Mean sqare error etc to compare order
order = [2,3,4,5,6,7,8,9,10]
colors = ['blue','black','red','green']
eval_x = [mse_Plot, mae_Plot, rtmse_Plot, variance_Plot]
for count in range(0,4):
    matplot.plot(order, eval_x[count], color=colors[count]) 
    matplot.legend(['MSE', 'MAE', 'Root MSE', 'Variance'])
    matplot.xlabel('Order')
    matplot.ylabel('Evaluate')
matplot.show()






