from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_curve, \
    roc_auc_score


def read_file():
    return np.loadtxt('data_multivar_nb.txt', delimiter=',')


def get_data():
    data = read_file()
    return data[:, :-1], data[:, -1]


def report(a, b, avg):
    texts = ["Precision", "Recall", "F1"]
    funcs = [precision_score, recall_score, f1_score]
    for i in range(len(texts)):
        print(texts[i], funcs[i](a, b, average=avg))
    print('')


x, y = get_data()
X_train, X_test, y_train, y_test = train_test_split(x, y)
svm_classifier = SVC(kernel='linear', probability=True)
svm_classifier.fit(X_train, y_train)
y_pred = svm_classifier.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
report(y_test, y_pred, "weighted")
y_prob = svm_classifier.predict_proba(X_test)
print("Roc auc score", roc_auc_score(y_test, y_prob, multi_class="ovr"))
