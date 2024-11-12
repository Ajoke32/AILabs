from pandas import read_csv
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn import preprocessing

names = ['age', 'work-class', 'final-weight',
         'education', 'years-of-education', 'marital-status',
         'occupation', 'relationship', 'race',
         'sex', 'capital-gain', 'capital-loss',
         'hours-per-week', 'native-country', 'income'
         ]

dataset = read_csv('income_data.txt', names=names)
models = [('LR', OneVsRestClassifier(LogisticRegression())),
          ('LDA', LinearDiscriminantAnalysis()),
          ('KNN', KNeighborsClassifier()),
          ('CART', DecisionTreeClassifier()),
          ('NB', GaussianNB()),
          ('SVM', SVC(gamma='auto'))
          ]

ignore = ['LDA', 'KNN', 'CART', 'NB', 'SVM']
array = dataset.values


def encode():
    X_encoded = np.empty(array.shape)
    label_encoders = []
    for i in range(array.shape[1] - 1):
        column = array[:, i]
        if str(column).isdigit():
            X_encoded[:, i] = column
        else:
            le = preprocessing.LabelEncoder()
            X_encoded[:, i] = le.fit_transform(column)
            label_encoders.append(le)

    return (X_encoded[:, :-1].astype(int), array[:, -1])


def scale(X, y):
    X_train, X_validation, Y_train, Y_validation = train_test_split(X, y, test_size=0.20, random_state=1)
    scaler = preprocessing.StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_validation = scaler.transform(X_validation)
    return (X_train, X_validation, Y_train, Y_validation)


X, y = encode()
X_train, X_validation, Y_train, Y_validation = scale(X, y)
reports = []
for name, model in models:
    model.fit(X_train, Y_train)
    predictions = model.predict(X_validation)
    report = classification_report(Y_validation, predictions)
    print(name)
    print(report)
