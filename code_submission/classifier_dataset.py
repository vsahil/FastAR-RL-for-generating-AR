import sys, os
import numpy as np
import pandas as pd
import sklearn
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.metrics import confusion_matrix
import random


def predict_single(arr, scaler, clf, pass_scaler=True):
    if isinstance(arr, tuple) or isinstance(arr, list):
        arr = np.array(arr)
    if pass_scaler:
        arr = scaler.transform(arr.reshape(1, -1))
    else:
        arr = arr.reshape(1, -1)
    return clf.predict(arr)[0]


def architecture(parameter, dataset, y, drop_, X, random_state, drop=True):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=random_state)
    scaler = MinMaxScaler(feature_range=(-1, 1))
    scaler.fit(X_train)
    X_train_ = scaler.transform(X_train)
    X_test_ = scaler.transform(X_test)
    if parameter == 3:      # used in dice-gradient. 
        return dataset, scaler, X_test, X_train, y_train, y_test
    # np.random.seed(42)
    # random.seed(42)
    clf = MLPClassifier(solver='lbfgs', alpha=1e-5,
                        hidden_layer_sizes=(5, 3), max_iter=10000, random_state=random_state)
    clf.fit(X_train_, y_train)

    if parameter:
        if drop:
            dataset = dataset.drop(columns=[*drop_])
        if parameter in [1, "policy_iteration"]:
            return clf, dataset, scaler, X_test, X_train
        elif parameter == 2:    # This is used in dice-gradient.py
            return clf, dataset, scaler, X_test, X_train, y_train, y_test
        else:
            raise ValueError("Parameter must be 1 or 2")
    Y_test_pred = clf.predict(X_test_)
    Y_train_pred = clf.predict(X_train_)
    tn, fp, fn, tp = confusion_matrix(y_test, Y_test_pred).ravel()
    tn, fp, fn, tp = confusion_matrix(y_train, Y_train_pred).ravel()
    print(random_state, clf.score(X_test_, y_test))
    print(random_state, (Y_test_pred == 0).sum(), (Y_test_pred == 0).sum() + (Y_train_pred == 0).sum())


def train_model_german(file=None, parameter=None, drop=True):
    if file == None:
        file = f"{os.path.dirname(os.path.realpath(__file__))}/datasets/german_redone.csv"
        # file = "datasets/german_redone.csv"
    dataset = pd.read_csv(file)
    y = dataset['target']
    if parameter == "policy_iteration":
        drop_ = ['target','Months', 'Purpose','Other-debtors','Other-installment-plans','Housing','Telephone','Present-employment-since','Present-residence-since','Property','Savings-account','Number-of-existing-credits','Insatllment-rate','Foreign-worker','Checking-account']
    else:
        drop_ = ['target']
    X = dataset.drop(columns=[*drop_])
    if "german_redone.csv" in file:
        random_state = 26
    else:
        random_state = int(sys.argv[1])

    return architecture(parameter, dataset, y, drop_, X, random_state, drop)


def train_model_adult(file=None, parameter=None, drop=True):
    if file == None:
        file = f"{os.path.dirname(os.path.realpath(__file__))}/datasets/adult_redone.csv"
    dataset = pd.read_csv(file)
    y = dataset['target']
    drop_ = ['target']
    X = dataset.drop(columns=[*drop_])
    if "adult_redone.csv"  in file:
        random_state = 50
    else:
        random_state = int(sys.argv[1])
    return architecture(parameter, dataset, y, drop_, X, random_state, drop)


def train_model_default(file=None, parameter=None, drop=True):
    if file == None:
        file = f"{os.path.dirname(os.path.realpath(__file__))}/datasets/default_redone.csv"
    dataset = pd.read_csv(file)
    y = dataset['target']
    drop_ = ['target']
    X = dataset.drop(columns=[*drop_])
    if "default_redone.csv" in file:
        random_state = 32
    else:
        random_state = int(sys.argv[1])

    return architecture(parameter, dataset, y, drop_, X, random_state, drop)


if __name__ == "__main__":
    train_model_german()
    train_model_adult()
    train_model_default()
