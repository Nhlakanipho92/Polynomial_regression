# The data cars give the speed of cars and the distances taken to stop (Note that the data were recorded in the 1920s).
# Date collected from http://sia.webpopix.org/polynomialRegression1.html
import numpy as np
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

# Training set
X_train = [[4], [4], [7], [7], [8], [9]] # Speed of car.
y_train = [[2], [10], [4], [22], [16], [10]] # Distance of car.

# Testing set
X_test = [[4], [4], [8], [9]] # speed
y_test = [[2], [4], [16], [10]] # Distance

# Train the Linear Regression model and plot a prediction
regressor = LinearRegression()
regressor.fit(X_train, y_train)
xx = np.linspace(0, 26, 100)
yy = regressor.predict(xx.reshape(xx.shape[0], 1))
plt.plot(xx, yy)

# Set the degree of the Polynomial Regression model
quadratic_featurizer = PolynomialFeatures(degree=2)

# This preprocessor transforms an input data matrix into a new data matrix of a given degree
X_train_quadratic = quadratic_featurizer.fit_transform(X_train)
X_test_quadratic = quadratic_featurizer.transform(X_test)

# Train and test the regressor_quadratic model
regressor_quadratic = LinearRegression()
regressor_quadratic.fit(X_train_quadratic, y_train)
xx_quadratic = quadratic_featurizer.transform(xx.reshape(xx.shape[0], 1))

# Plot the graph
plt.plot(xx, regressor_quadratic.predict(xx_quadratic), c='r', linestyle='--')
plt.title('Speed of car and the distances taken to stop in 1920')
plt.xlabel('Speed')
plt.ylabel('Distance')
plt.axis([0, 25, 0, 25])
plt.grid(True)
plt.scatter(X_train, y_train)
plt.show()
print (X_train)
print (X_train_quadratic)
print (X_test)
print (X_test_quadratic)