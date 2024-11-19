import matplotlib.pyplot as plt
from sklearn import datasets, linear_model
from sklearn.model_selection import train_test_split
import sklearn.metrics as sm

diabetes = datasets.load_diabetes()
X = diabetes.data
y = diabetes.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=0)
regr = linear_model.LinearRegression()
regr.fit(X_train, y_train)
y_test_pred = regr.predict(X_test)


def print_info():
    print("Linear regressor performance:")
    print(F"Coef: {regr.coef_}")
    print(F"Intercept: {regr.intercept_}")
    print("R2 score =", round(sm.r2_score(y_test, y_test_pred), 2))
    print("Mean absolute error =", round(sm.mean_absolute_error(y_test, y_test_pred), 2))
    print("Mean squared error =", round(sm.mean_squared_error(y_test, y_test_pred), 2))


def plot():
    fig, ax = plt.subplots()
    ax.scatter(y_test, y_test_pred, edgecolors=(0, 0, 0))
    ax.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', lw=4)
    ax.set_xlabel('Виміряно')
    ax.set_ylabel('Передбачено')
    plt.show()

print_info()
#plot()