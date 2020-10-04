import numpy as np
import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import sys
# from sklearn.metrics import accuracy_score

def predict_single(arr, scaler, clf):
    arr = scaler.transform(arr.reshape(1, -1))
    print(clf.predict(arr)[0])

def train_model(parameter=None):
    # 2 or 5 categories are good 
    file = "german_redone_2.csv"
    dataset = pd.read_csv(file)
    # import ipdb; ipdb.set_trace()
    y = dataset['target']
    drop_ = ['target','Purpose','Other-debtors','Other-installment-plans','Housing','Telephone','Present-employment-since','Present-residence-since','Property','Savings-account','Number-of-existing-credits','Insatllment-rate']
    X = dataset.drop(columns=[*drop_])
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=35) #random_state=int(sys.argv[1]))

    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train_ = scaler.transform(X_train)
    X_test_ = scaler.transform(X_test)

    clf = MLPClassifier(solver='lbfgs', alpha=1e-5,
                        hidden_layer_sizes=(5, 3), max_iter=10000, random_state=1)

    clf.fit(X_train_, y_train)
    if parameter:
        return clf, dataset.drop(columns=[*drop_]), scaler

    Y_test_pred = clf.predict(X_test)
    print(clf.score(X_test, y_test))
    # print(accuracy_score(y_test, Y_test_pred))
    # arr = np.array([4,0,4,3,0,5,3,1,3,1,2,1,1,3,2,2,2,2,1,1])
    # predict_single(arr, scaler, clf)

if __name__ == "__main__":
    train_model()
