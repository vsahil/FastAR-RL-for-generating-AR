import numpy as np
import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import sys
from sklearn.metrics import confusion_matrix

# from sklearn.metrics import accuracy_score

def predict_single(arr, scaler, clf):
    if isinstance(arr, tuple) or isinstance(arr, list):
        arr = np.array(arr)
    arr = scaler.transform(arr.reshape(1, -1))
    return clf.predict(arr)[0]

def train_model(parameter=None):
    # 2 or 5 categories are good 
    file = "german_redone_4.csv"    # 4 is also good
    dataset = pd.read_csv(file)
    # import ipdb; ipdb.set_trace()
    y = dataset['target']
    drop_ = ['target','Months','Purpose','Other-debtors','Other-installment-plans','Housing','Telephone','Present-employment-since','Present-residence-since','Property','Savings-account','Number-of-existing-credits','Insatllment-rate','Foreign-worker','Checking-account']
    # 'Credit-history', 'Credit-amount', 'Personal-status', 'age', 'Job', 'Number-of-people-being-lible'
    X = dataset.drop(columns=[*drop_])
    random_state = 7
    # random_state=int(sys.argv[1]))
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=random_state)

    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train_ = scaler.transform(X_train)
    X_test_ = scaler.transform(X_test)

    clf = MLPClassifier(solver='lbfgs', alpha=1e-5,
                        hidden_layer_sizes=(5, 3), max_iter=10000, random_state=1)

    clf.fit(X_train_, y_train)
    if parameter:
        return clf, dataset.drop(columns=[*drop_]), scaler, X_test

    Y_test_pred = clf.predict(X_test_)     # this should not be X_test, was a bug. 
    import ipdb; ipdb.set_trace()
    tn, fp, fn, tp = confusion_matrix(y_test, Y_test_pred).ravel()
    for no, i in enumerate(X_test.to_numpy()):
        if predict_single(i, scaler, clf) == 0:
            print(no, tuple(i))
    print(clf.score(X_test, y_test), random_state)
    # print(accuracy_score(y_test, Y_test_pred))
    # arr = np.array([4,0,4,3,0,5,3,1,3,1,2,1,1,3,2,2,2,2,1,1])
    # predict_single(np.array([4, 3, 1, 2, 4.5, 1.0]), scaler, clf)

if __name__ == "__main__":
    train_model()
