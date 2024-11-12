from pandas import read_csv
from pandas.plotting import scatter_matrix
from matplotlib import pyplot
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
import numpy as np

url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/iris.csv"
names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']
dataset = read_csv(url, names=names)


def print_info():
    print(dataset.shape)
    print(dataset.head(5))
    print(dataset.describe())
    print(dataset.groupby('class').size())


def build_diagram():
    dataset.plot(kind='box', subplots=True, layout=(2, 2), sharex=False, sharey=False)
    pyplot.show()


def build_histogram():
    dataset.hist()
    pyplot.show()


def build_scatter_matrix():
    scatter_matrix(dataset)
    pyplot.show()

def split_data():
    array = dataset.values
    X = array[:, 0:4]
    y = array[:, 4]
    X_train, X_validation, Y_train, Y_validation = train_test_split(X, y, test_size=0.20, random_state=1)
    return (X_train, X_validation, Y_train, Y_validation)


def build_and_rate_models(print_all=False):
    models = []
    models.append(('LR', OneVsRestClassifier(LogisticRegression())))
    models.append(('LDA', LinearDiscriminantAnalysis()))
    models.append(('KNN', KNeighborsClassifier()))
    models.append(('CART', DecisionTreeClassifier()))
    models.append(('NB', GaussianNB()))
    models.append(('SVM', SVC(gamma='auto')))
    results = []
    names = []
    X_train, X_validation, Y_train, Y_validation = split_data()
    for name, model in models:
        kfold = StratifiedKFold(n_splits=10, random_state=1, shuffle=True)
        cv_results = cross_val_score(model, X_train, Y_train, cv=kfold, scoring='accuracy')
        results.append(cv_results)
        names.append(name)
        predictions = predict(model, X_train, Y_train, X_validation)
        if print_all:
            print(F"Predictions {name}: \n{predictions}")
            rate_prediction(Y_validation, predictions)
            print('%s: %f (%f)' % (name, cv_results.mean(), cv_results.std()))
            print(F"{name} \n {predict(model, X_train, Y_train, np.array([[4, 2.9, 1, 0.2]]), True)}")

    #pyplot.boxplot(results, tick_labels=names)
    #pyplot.title('Algorithm Comparison')
    #pyplot.show()

def predict(model, X_train, Y_train, X_validation, with_label = False):
    model.fit(X_train, Y_train)
    predictions = model.predict(X_validation)
    if with_label:
        print(F"prediction label: {dataset['class'][0]}")
    return predictions


def rate_prediction(Y_validation, predictions):
    print(accuracy_score(Y_validation, predictions))
    print(confusion_matrix(Y_validation, predictions))
    print(classification_report(Y_validation, predictions))

array = dataset.values
X = array[:, 0:4]
y = array[:, 4]
print(X, y)

