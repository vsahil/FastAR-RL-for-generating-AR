import pandas as pd
import numpy as np
from pandas.api.types import is_numeric_dtype
import copy


def convert_numerical_to_categorical(df):
    # numerical: Months, unique values = 33
    # numerical: Credit-amount, unique values = 921
    # numerical: age, unique values = 53
    # Split into 2, 3, 4, or 5 categories each of them.
    # Initial accuracy = 70.5% with all numerical variables. 
    # import ipdb; ipdb.set_trace()
    values = [2, 3, 4, 5]
    copies = []
    for _ in values:
        copies.append(copy.deepcopy(df))
    for k, nos_categories in enumerate(values):
        copies[k]['Months'] = pd.qcut(df['Months'], nos_categories, labels=False)
        copies[k]['Credit-amount'] = pd.qcut(df['Credit-amount'], nos_categories, labels=False)
        copies[k]['age'] = pd.qcut(df['age'], nos_categories, labels=False)
        copies[k].to_csv(f"german_redone_{nos_categories}.csv", index=False)


def raw_to_no_missing():
    o_df = pd.read_csv("original_german.csv")
    # remove rows with missing values
    df = o_df.dropna()
    assert df.shape[0] == o_df.shape[0]

    # Change all alphanumeric features into numerical categories
    df['Checking-account'] = df['Checking-account'].replace({'A11':1, 'A12':2, 'A13':3, 'A14':4})
    df['Credit-history'] = df['Credit-history'].replace({'A30':0, 'A31':1, 'A32':2, 'A33':3, 'A34':4, 'A35':5})
    df['Purpose'] = df['Purpose'].replace({'A40':0, 'A41':1, 'A42':2, 'A43':3, 'A44':4, 'A45':5, 'A46':6, 'A47':7, 'A48':8, 'A49':9, 'A410':10})
    df['Savings-account'] = df['Savings-account'].replace({'A61':1, 'A62':2, 'A63':3, 'A64':4, 'A65':5})
    df['Present-employment-since'] = df['Present-employment-since'].replace({'A71':1, 'A72':2, 'A73':3, 'A74':4, 'A75':5})
    # considers only sex now
    df['Personal-status'] = df['Personal-status'].replace({'A91':1, 'A92':0, 'A93':1, 'A94':1, 'A95':0})
    df['Other-debtors'] = df['Other-debtors'].replace({'A101':1, 'A102':2, 'A103':3})
    df['Property'] = df['Property'].replace({'A121':1, 'A122':2, 'A123':3, 'A124':4})
    df['Other-installment-plans'] = df['Other-installment-plans'].replace({'A141':1, 'A142':2, 'A143':3})
    df['Housing'] = df['Housing'].replace({'A151':1, 'A152':2, 'A153':3})
    df['Job'] = df['Job'].replace({'A171':1, 'A172':2, 'A173':3, 'A174':4})
    df['Telephone'] = df['Telephone'].replace({'A191':1, 'A192':2})
    df['Foreign-worker'] = df['Foreign-worker'].replace({'A201':1, 'A202':2})
    df['target'] = df['target'].replace({1:1, 2:0})

    # import ipdb; ipdb.set_trace()
    # for i in df.columns:
    #     # assert(isinstance(df[i].dtype, np.int64))
    #     print(i, df[i].dtype)
    #     assert is_numeric_dtype(df[i])

    # df_normalized = df.drop('target', axis=1)
    # df_normalized = df_normalized.apply(lambda x: (x - np.min(x)) / (np.max(x) - np.min(x)))        # corrected to min, instead of mean
    df.to_csv("german_redone.csv", index=False)
    convert_numerical_to_categorical(df)
    # df_normalized.to_csv("german_redone_normalized_withheader.csv", index=False)

import sys
if __name__ == "__main__":
    cleaning_level = int(sys.argv[1])
    # if cleaning_level == 0:
    #     really_original_to_original()
    if cleaning_level == 1:
        raw_to_no_missing()
    # elif cleaning_level == 2:
    #    missing_to_normalized() 
    # elif cleaning_level == 3:
    #     print_mins_and_ranges()
    # elif cleaning_level == 4:
    #     convert_to_nosensitive()
    # else:
    #     raise NotImplementedError
        
