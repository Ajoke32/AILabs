import numpy as np
from sklearn import preprocessing
from sklearn.svm import LinearSVC, SVC
from sklearn.multiclass import OneVsOneClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, recall_score, precision_score

input_file = 'income_data.txt'

max_datapoints = 25_000
with open(input_file, 'r') as f:
    X = []
    c1 = 0
    c2 = 0
    for line in f.readlines():
        if c1 >= max_datapoints and c2 >= max_datapoints:
            break
        if '?' in line:
            continue
        data = line[:-1].split(', ')
        if data[-1] == '<=50K' and c1 < max_datapoints:
            X.append(data)
            c1 += 1

        if data[-1] == '>50K' and c2 < max_datapoints:
            X.append(data)
            c2 += 1
    X = np.array(X)
    X_encoded = np.empty(X.shape)
    label_encoder = []
    for i, item in enumerate(X[0]):
        if str(item).isdigit():
            X_encoded[:, i] = X[:, i]
        else:
            label_encoder.append(preprocessing.LabelEncoder())
            X_encoded[:, i] = label_encoder[-1].fit_transform(X[:, i])
    X = X_encoded[:, :-1].astype(int)
    Y = X_encoded[:, -1].astype(int)
    classifier = OneVsOneClassifier(SVC(kernel='rbf'))
    classifier.fit(X, Y)
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=5)
    classifier = OneVsOneClassifier(SVC(kernel='rbf'))
    classifier.fit(X_train, Y_train)
    y_test_pred = classifier.predict(X_test)
    f1 = cross_val_score(classifier, X, Y, cv=3, scoring='precision_weighted')
    print("F1 score: " + str(round(100*f1.mean(), 2)) + "%")
    print(F"accuracy_score {accuracy_score(Y_test, y_test_pred):.2%}")
    print(F"recall_score {recall_score(Y_test, y_test_pred, average='weighted'):.2%}")
    print(F"precision_score {precision_score(Y_test, y_test_pred, average='weighted'):.2%}")
    input_data = ['37', 'Private', '215646', 'HS-grad', '9', 'Never-married', 'Handlers-cleaners', 'Not-in-family',
                      'White', 'Male',
                      '0', '0', '40', 'United-States']
    input_data_encoded = [-1] * len(input_data)
    count = 0
    for i, item in enumerate(input_data):
        if item.isdigit():
            input_data_encoded[i] = int(input_data[i])
        else:
            input_data_encoded[i] = int(label_encoder[count].transform([input_data[i]])[0])
            count += 1
    input_data_encoded = np.array(input_data_encoded).reshape(1, 14)
    predicted_class = classifier.predict(input_data_encoded)
    print(label_encoder[-1].inverse_transform(predicted_class)[0])