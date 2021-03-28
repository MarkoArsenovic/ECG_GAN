from sklearn.svm import SVC, LinearSVC
from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import HalvingGridSearchCV
from sklearn.model_selection import StratifiedKFold
from load_dataset import load_train_test_ECG_dateset

X_axis, Y_axis =  load_train_test_ECG_dateset()
print("DataSet has been loaded")

skf = StratifiedKFold(n_splits=5, random_state=None, shuffle=False)
print(skf)
X_train = []
X_test = []
y_train = []
y_test = []

for train_index, test_index in skf.split(X_axis, Y_axis):
    print("TRAIN:", train_index, "TEST:", test_index)
    print()

    for i in train_index:
        X_train.append(X_axis[i]) 
        y_train.append(Y_axis[i])

    for j in test_index:
        X_test.append(X_axis[j]) 
        y_test.append(Y_axis[j])


parameters = {"class_weight":[{0:10, 1:8, 2:4, 3:6, 4:7, 5:7},
                              {0:4, 1:9, 2:0.12, 3:14, 4:10, 5:10},
                              "balanced"],
             "C" : [4, 8]}

'''
parameters = {"C" : [1.0,1.2, 1.5, 2, 4, 8, 10]}  

parameters = {"class_weight":[{'N':0.00001, 'L':0.001, 'R':0.001, 'A':1, 'V':10, '/':10},
                            {'N':0.0001, 'L':0.002, 'R':0.001, 'A':1, 'V':10, '/':10},
                            "balanced"],
             "gamma":["auto"],
             "C" : [0.1,0.15,0.2,0.25,0.3,0.5,0.7,1.0,1.5,2,5],
             "kernel":["linear","rbf","poly"]}
'''

#grid = HalvingGridSearchCV(SVC(),parameters, refit=True,cv=skf, scoring="balanced_accuracy",n_jobs=-1)
#grid = HalvingGridSearchCV(LinearSVC(), parameters, refit=True,cv=skf, scoring="balanced_accuracy",n_jobs=-1)
grid = HalvingGridSearchCV(LinearSVC(), parameters, refit=True,cv=skf, scoring="balanced_accuracy",n_jobs=-1)

#grid = LinearSVC()

X_train

grid.fit(X_train, y_train)
"""
parameters = {"class_weight":[{1:0.1,2:10},{1:0.11,2:10},{1:0.116,2:10},{1:0.115,2:10},
                              {1:0.000048,2:10},{1:0.00006,2:10},{1:0.02,2:10},{1:0.02,2:10},{1:0.015,2:10},{1:0.027,2:10},
                              {1:0.015,2:10},{1:0.025,2:10},{1:0.03,2:10},{1:0.02,2:20},{1:0.018,2:10},{1:0.023,2:10},
                              {1:0.00006,2:10},{1:0.000062,2:10},{1:0.00005,2:10},{1:0.000053,2:10},{1:0.000057,2:10},
                              {1:0.00004,2:10},{1:0.1165,2:10},{1:0.114,2:10},{1:0.113,2:10},{1:1.15,2:10},"balanced"],
             "gamma":["auto"],
             "C" : [0.1,0.15,0.2,0.25,0.3,0.5,0.7,1.0,1.5,2,5],
             "kernel":["linear","rbf","poly"]} #"sigmoid" not a good one
grid = HalvingGridSearchCV(SVC(),parameters, refit=True,cv=skf, scoring="balanced_accuracy",n_jobs=-1)
grid.fit(train,validate)
"""