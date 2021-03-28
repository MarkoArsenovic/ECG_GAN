from sklearn.svm import SVC, LinearSVC, NuSVC
from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import HalvingGridSearchCV 
from sklearn.model_selection import StratifiedKFold
from load_dataset import load_train_test_ECG_dateset
from sklearn.metrics import accuracy_score

from sklearn.gaussian_process import GaussianProcessClassifier

    #One-Vs-The-Rest

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV, SGDClassifier, Perceptron, PassiveAggressiveClassifier
import numpy as np

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


X_train = np.array(X_train)
X_test = np.array(X_test)
y_train = np.array(y_train)
y_test = np.array(y_test)


def train_and_predict(model, X_train, X_test, y_train, y_test):
    model.fit(X_train, y_train) 
    test_pred = model.predict(X_test) 
    print(f"Test Set Accuracy : {accuracy_score(y_test, test_pred) * 100} %\n\n")  

choose_methode = "Perceptron"

if choose_methode ==  "PassiveAggressiveClassifier" or True: #  67.07951871853324 %
    #model = PassiveAggressiveClassifier(C = 0.5, random_state = 5) 
    #train_and_predict(model, X_train, X_test, y_train, y_test)
    pass
if choose_methode == "Perceptron" or True: # 53.49441064963245 %
    #model = Perceptron()
    #train_and_predict(model, X_train, X_test, y_train, y_test)
    pass
if choose_methode == "SGDClassifier" or True:   #84.50585754166107 %
    #model = SGDClassifier() 
    #train_and_predict(model, X_train, X_test, y_train, y_test)
    pass
if choose_methode == "LogisticRegressionCV" or True:
    # Failed to converge
    # ConvergenceWarning: lbfgs failed to converge (status=1): STOP: TOTAL NO. of ITERATIONS REACHED LIMIT.
    # Increase the number of iterations (max_iter) or scale the data as shown in:
    #     https://scikit-learn.org/stable/modules/preprocessing.html
    # Please also refer to the documentation for alternative solver options:
    #     https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression
    #model = LogisticRegressionCV(multi_class='ovr')
    #train_and_predict(model, X_train, X_test, y_train, y_test)
    pass
if choose_methode == "LogisticRegression" or True: # 83.7232265850715 %
    #model = LogisticRegression(multi_class='ovr')
    #train_and_predict(model, X_train, X_test, y_train, y_test)
    pass
if choose_methode == "LinearSVC" or True: # 77.48655995903988 %
    #model = LinearSVC(multi_class='ovr')
    #train_and_predict(model, X_train, X_test, y_train, y_test)
    pass
if choose_methode == "GaussianProcessClassifier" or True: 
    # Memory problem
    #model = GaussianProcessClassifier(multi_class = 'one_vs_rest')
    #train_and_predict(model, X_train, X_test, y_train, y_test)
    pass
if choose_methode == "GradientBoostingClassifier" or True:
    model = GradientBoostingClassifier()
    train_and_predict(model, X_train, X_test, y_train, y_test)




"""

    #One-Vs-One

svm.NuSVC
svm.SVC.
gaussian_process.GaussianProcessClassifier (setting multi_class = “one_vs_one”)


    #One-Vs-The-Rest

ensemble.GradientBoostingClassifier
gaussian_process.GaussianProcessClassifier (setting multi_class = “one_vs_rest”)
svm.LinearSVC (setting multi_class=”ovr”)
linear_model.LogisticRegression (setting multi_class=”ovr”)
linear_model.LogisticRegressionCV (setting multi_class=”ovr”)
linear_model.SGDClassifier
linear_model.Perceptron
linear_model.PassiveAggressiveClassifier

"""