import numpy as np
import matplotlib.pyplot as plt
from sklearn.naive_bayes import GaussianNB
import sklearn.model_selection as train_test_split
from utilities import visualize_classifier
from sklearn.metrics import f1_score

input_file = 'data_multivar_nb.txt'
data = np.loadtxt(input_file, delimiter=',')
x, y = data[:, :-1], data[:, -1]
classifier = GaussianNB()
classifier.fit(x, y)
y_pred = classifier.predict(x)
accuracy = 100.0 * (y == y_pred).sum() / x.shape[0]
#print("Accuracy of Naive Bayes classifier =", round(accuracy, 2), "%")
#visualize_classifier(classifier, x, y)


x_train, X_test, y_train, y_test = train_test_split.train_test_split(x, y, test_size=0.8, random_state=3)
classifier_new = GaussianNB()
classifier_new.fit(x_train, y_train)
y_test_pred = classifier_new.predict(X_test)
#accuracy = 100.0 * (y_test == y_test_pred).sum() / X_test.shape[0]
#print("Accuracy of the new classifier =", round(accuracy, 2), "%")
#visualize_classifier(classifier_new, X_test, y_test)

num_folds = 3
accuracy_values = train_test_split.cross_val_score(classifier, x, y, scoring='accuracy', cv=num_folds)
print("Accuracy: " + str(round(100 * accuracy_values.mean(), 2)) + "%")
precision_values = train_test_split.cross_val_score(classifier, x, y, scoring='precision_weighted', cv=num_folds)
print("Precision: " + str(round(100 * precision_values.mean(), 2)) + "%")
recall_values = train_test_split.cross_val_score(classifier, x, y, scoring='recall_weighted', cv=num_folds)
print("Recall: " + str(round(100 * recall_values.mean(), 2)) + "%")
f1_values = train_test_split.cross_val_score(classifier_new, x, y, scoring='f1_weighted', cv=num_folds)
print("F1: " + str(round(100 * f1_values.mean(), 2)) + "%")
