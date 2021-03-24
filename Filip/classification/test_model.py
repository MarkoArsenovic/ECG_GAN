import joblib
from load_dataset import load_train_test_ECG_dateset
from sklearn.model_selection import StratifiedKFold


X_axis, Y_axis =  load_train_test_ECG_dateset()

skf = StratifiedKFold(n_splits=5, random_state=None, shuffle=False)

X_train = []
X_test = []
y_train = []
y_test = []


for train_index, test_index in skf.split(X_axis, Y_axis):
    for i in train_index:
        X_train.append(X_axis[i]) 
        y_train.append(Y_axis[i])

    for j in test_index:
        X_test.append(X_axis[j]) 
        y_test.append(Y_axis[j])

loaded_model = joblib.load('models/svm.sav')



y_pred = loaded_model.predict(X_test)

#Confusion Matrix
from sklearn.metrics import confusion_matrix
print(confusion_matrix(y_test, y_pred))

# Accuracy
from sklearn.metrics import accuracy_score
print(accuracy_score(y_test, y_pred))

# Recall
from sklearn.metrics import recall_score
print(recall_score(y_test, y_pred, average=None))

# Precision
from sklearn.metrics import precision_score
print(precision_score(y_test, y_pred, average=None))