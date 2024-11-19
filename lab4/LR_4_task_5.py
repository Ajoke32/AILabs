import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn.preprocessing import PolynomialFeatures
import sklearn.metrics as sm

m = 100
X = 6 * np.random.rand(m, 1) - 3
y = 0.6 * X ** 2 + X + 2 + np.random.randn(m, 1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=0)


def show_data():
    plt.figure(figsize=(10, 6))
    plt.scatter(X_train, y_train, color='blue', label='Train Data')
    plt.scatter(X_test, y_test, color='red', label='Test Data')
    plt.title('Генерація та розділення даних')
    plt.xlabel('X')
    plt.ylabel('y')
    plt.legend()
    plt.grid(True)
    plt.show()


def linear():
    regr = linear_model.LinearRegression()
    regr.fit(X_train, y_train)
    y_test_pred = regr.predict(X_test)
    fig, ax = plt.subplots()
    ax.scatter(y_test, y_test_pred, edgecolors=(0, 0, 0))
    ax.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', lw=4)
    plt.title('linear')
    plt.show()


def poly_linear():
    poly_features = PolynomialFeatures(degree=2, include_bias=False)
    X_poly = poly_features.fit_transform(X)
    lin_reg = linear_model.LinearRegression()
    lin_reg.fit(X_poly, y)
    X_test_poly = poly_features.transform(X_test)
    y_test_pred = lin_reg.predict(X_test_poly)
    X_plot = np.linspace(-3, 3, 300).reshape(-1, 1)
    X_plot_poly = poly_features.transform(X_plot)
    y_plot = lin_reg.predict(X_plot_poly)
    plt.figure(figsize=(10, 6))
    plt.scatter(X, y, color='blue', label="Data")
    plt.plot(X_plot, y_plot, color="red", linewidth=2, label='Regression line')
    plt.title("Поліноміальна регресія (ступінь 2)")
    plt.xlabel('X')
    plt.ylabel('y')
    plt.legend()
    plt.grid(True)
    plt.show()
    rate(y_test, y_test_pred)
    coefficients = lin_reg.coef_
    intercept = lin_reg.intercept_
    print(f"y = {coefficients[0][1]:.2f} * x^2 + {coefficients[0][0]:.2f} * x + {intercept[0]:.2f}")


def rate(test, pred):
    print("Poly linear regressor performance:")
    print("Mean absolute error =", round(sm.mean_absolute_error(test, pred), 2))
    print("Mean squared error =", round(sm.mean_squared_error(test, pred), 2))
    print("Median absolute error =", round(sm.median_absolute_error(test, pred), 2))
    print("Explain variance score =", round(sm.explained_variance_score(test, pred), 2))
    print("R2 score =", round(sm.r2_score(test, pred), 2))


poly_linear()
linear()
show_data()
