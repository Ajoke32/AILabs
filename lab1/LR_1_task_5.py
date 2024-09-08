import pandas as pd
from sklearn.metrics import confusion_matrix, accuracy_score, recall_score, precision_score, f1_score, roc_curve, \
    roc_auc_score
import numpy as np
import matplotlib.pyplot as plt


def find_TP(y_true, y_pred):
    return sum((y_true == 1) & (y_pred == 1))


def find_FN(y_true, y_pred):  # counts the number of false negatives (y_true = 1, y_pred = 0)
    return sum((y_true == 1) & (y_pred == 0))


def find_FP(y_true, y_pred):  # counts the number of false positives (y_true = 0, y_pred = 1)
    return sum((y_true == 0) & (y_pred == 1))


def find_TN(y_true, y_pred):  # counts the number of true negatives (y_true = 0, y_pred = 0)
    return sum((y_true == 0) & (y_pred == 0))


def find_conf_matrix_values(y_true, y_pred):
    # calculate TP, FN, FP, TN
    TP = find_TP(y_true, y_pred)
    FN = find_FN(y_true, y_pred)
    FP = find_FP(y_true, y_pred)
    TN = find_TN(y_true, y_pred)
    return TP, FN, FP, TN


def bekker_confusion_matrix(y_true, y_pred):
    TP, FN, FP, TN = find_conf_matrix_values(y_true, y_pred)
    return np.array([[TN, FP], [FN, TP]])


def bekker_accuracy_score(y_true, y_pred):
    TP, FN, FP, TN = find_conf_matrix_values(y_true, y_pred)
    return (TP + TN) / (FN + FP + TN + TP)


def bekker_recall_score(y_true, y_pred):
    # calculates the fraction of positive samples predicted correctly
    TP, FN, FP, TN = find_conf_matrix_values(y_true, y_pred)
    return TP / (TP + FN)


def bekker_precision_score(y_true, y_pred):
    # calculates the fraction of predicted positives samples that are actually positive
    TP, FN, FP, TN = find_conf_matrix_values(y_true, y_pred)
    return TP / (TP + FP)


def bekker_f1_score(y_true, y_pred):
    # calculates the F1 score
    recall = bekker_recall_score(y_true, y_pred)
    precision = bekker_precision_score(y_true, y_pred)
    return 2 * (precision * recall) / (precision + recall)


df = pd.read_csv('data_metrics.csv')
df.head()

thresh = 0.5
df['predicted_RF'] = (df.model_RF >= 0.5).astype('int')
df['predicted_LR'] = (df.model_LR >= 0.5).astype('int')
df.head()

matrix = confusion_matrix(df.actual_label.values, df.predicted_RF.values)
print("actual:", matrix)

print("my:", bekker_confusion_matrix(df.actual_label.values, df.predicted_RF.values))

assert np.array_equal(bekker_confusion_matrix(df.actual_label.values, df.predicted_RF.values),
                      confusion_matrix(df.actual_label.values, df.predicted_RF.values)
                      ), 'bekker_confusion_matrix() is not correct for RF'
assert np.array_equal(bekker_confusion_matrix(df.actual_label.values, df.predicted_LR.values),
                      confusion_matrix(df.actual_label.values, df.predicted_LR.values)
                      ), 'bekker_confusion_matrix() is not correct for LR'


def eq(val, other):
    return val == other


assert eq(bekker_accuracy_score(df.actual_label.values, df.predicted_RF.values),
          accuracy_score(df.actual_label.values, df.predicted_RF.values)), 'bekker_accuracy_score failed on RF'

assert eq(bekker_accuracy_score(df.actual_label.values, df.predicted_LR.values),
          accuracy_score(df.actual_label.values, df.predicted_LR.values)), 'bekker_accuracy_score failed on LR'

print('Accuracy RF: %.3f' % (bekker_accuracy_score(df.actual_label.values, df.predicted_RF.values)))
print('Accuracy LR: %.3f' % (bekker_accuracy_score(df.actual_label.values, df.predicted_LR.values)))

assert eq(bekker_recall_score(df.actual_label.values, df.predicted_RF.values),
          recall_score(df.actual_label.values, df.predicted_RF.values)), 'bekker_recall_score failed on RF'

assert eq(bekker_recall_score(df.actual_label.values, df.predicted_LR.values),
          recall_score(df.actual_label.values, df.predicted_LR.values)), 'bekker_recall_score failed on LR'

print('Recall RF: %.3f' % (bekker_recall_score(df.actual_label.values, df.predicted_RF.values)))
print('Recall LR: %.3f' % (bekker_recall_score(df.actual_label.values, df.predicted_LR.values)))

assert eq(bekker_precision_score(df.actual_label.values, df.predicted_RF.values),
          precision_score(df.actual_label.values, df.predicted_RF.values)), 'bekker_precision_score failed on RF'

assert eq(bekker_precision_score(df.actual_label.values, df.predicted_LR.values),
          precision_score(df.actual_label.values, df.predicted_LR.values)), 'bekker_accuracy_score failed on LR'

print('Precision RF: %.3f' % (bekker_precision_score(df.actual_label.values, df.predicted_RF.values)))
print('Precision LR: %.3f' % (bekker_precision_score(df.actual_label.values, df.predicted_LR.values)))

assert eq(bekker_f1_score(df.actual_label.values, df.predicted_RF.values),
          f1_score(df.actual_label.values, df.predicted_RF.values)), 'bekker_f1_score failed on RF'

assert eq(round(bekker_f1_score(df.actual_label.values, df.predicted_LR.values), 3),
          round(f1_score(df.actual_label.values, df.predicted_LR.values), 3)), 'bekker_f1_score failed on LR'

print('F1 RF: %.3f' % (bekker_f1_score(df.actual_label.values, df.predicted_RF.values)))
print('F1 LR: %.3f' % (bekker_f1_score(df.actual_label.values, df.predicted_LR.values)))

print('scores with threshold = 0.5')
print('Accuracy RF: %.3f' % (bekker_accuracy_score(df.actual_label.values, df.predicted_RF.values)))
print('Recall RF: %.3f' % (bekker_recall_score(df.actual_label.values, df.predicted_RF.values)))
print('Precision RF: %.3f' % (bekker_precision_score(df.actual_label.values, df.predicted_RF.values)))
print('F1 RF: %.3f' % (bekker_f1_score(df.actual_label.values, df.predicted_RF.values)))
print('')
print('scores with threshold = 0.25')
print('Accuracy RF: %.3f' % (bekker_accuracy_score(df.actual_label.values, (df.model_RF >= 0.25).astype('int').values)))
print('Recall RF: %.3f' % (bekker_recall_score(df.actual_label.values, (df.model_RF >= 0.25).astype('int').values)))
print(
    'Precision RF: %.3f' % (bekker_precision_score(df.actual_label.values, (df.model_RF >= 0.25).astype('int').values)))
print('F1 RF: %.3f' % (bekker_f1_score(df.actual_label.values, (df.model_RF >= 0.25).astype('int').values)))


def show(p):
    p.plot([0, 1], [0, 1], 'k-', label='random')
    p.plot([0, 0, 1, 1], [0, 1, 1, 1], 'g-', label='perfect')
    p.legend()
    p.xlabel('False Positive Rate')
    p.ylabel('True Positive Rate')
    p.show()


fpr_RF, tpr_RF, thresholds_RF = roc_curve(df.actual_label.values, df.model_RF.values)
fpr_LR, tpr_LR, thresholds_LR = roc_curve(df.actual_label.values, df.model_LR.values)
plt.plot(fpr_RF, tpr_RF, 'r-', label='RF')
plt.plot(fpr_LR, tpr_LR, 'b-', label='LR')
show(plt)

auc_RF = roc_auc_score(df.actual_label.values, df.model_RF.values)
auc_LR = roc_auc_score(df.actual_label.values, df.model_LR.values)
print('AUC RF:%.3f' % auc_RF)
print('AUC LR:%.3f' % auc_LR)
plt.plot(fpr_RF, tpr_RF, 'r-', label='RF AUC: %.3f' % auc_RF)
plt.plot(fpr_LR, tpr_LR, 'b-', label='LR AUC: %.3f' % auc_LR)
show(plt)
