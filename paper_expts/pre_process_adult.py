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
    
    o_df = pd.read_csv("adult_train.csv")
    # remove rows with missing values
    
    df = o_df.dropna()
    # print(df.shape, o_df.shape)
    assert df.shape[0] == 45222     

    for i in range(1, 17):
        assert(len(df.loc[df['education-num'] == i, 'education'].value_counts().index.tolist()) == 1)        # this means education-num and education have one-to one mapping, can drop one of them
    df = df.drop('education-num', axis=1)

    # Change all categorical features into numeric
    df.workclass, mapping_index = pd.Series(df.workclass).factorize()
    # Preschool < 1st-4th < 5th-6th < 7th-8th < 9th < 10th < 11th < 12th < HS-grad < Prof-school < Assoc-acdm < Assoc-voc < Some-college < Bachelors < Masters < Doctorate (https://www.rdocumentation.org/packages/arules/versions/1.6-6/topics/Adult)
    education_map = {' Preschool':0, ' 1st-4th':1, ' 5th-6th':2,  ' 7th-8th':3, ' 9th':4, ' 10th':5, ' 11th':6, ' 12th':7, ' HS-grad':8, ' Prof-school':9, ' Assoc-acdm':10, ' Assoc-voc':11, ' Some-college':12, ' Bachelors':13, ' Masters':14, ' Doctorate':15}
    df['education'] = df['education'].replace(education_map)
    # df.education, mapping_index = pd.Series(df.education).factorize()
    df['marital-status'], mapping_index = pd.Series(df['marital-status']).factorize()
    df.occupation, mapping_index = pd.Series(df.occupation).factorize()
    df.relationship, mapping_index = pd.Series(df.relationship).factorize()
    df.race, mapping_index = pd.Series(df.race).factorize()
    df.sex, mapping_index = pd.Series(df.sex).factorize()
    df['native-country'], mapping_index = pd.Series(df['native-country']).factorize()
    # sex_map = {'Male':1, 'Female':0}
    # df['sex'] = df['sex'].replace(sex_map)
    outcome_map = {' >50K':1, ' >50K.':1, ' <=50K':0, ' <=50K.':0}
    df['target'] = df['target'].replace(outcome_map)
    # df = df.rename(columns={"class": "target"})
    for i in df.columns:
        # assert(isinstance(df[i].dtype, np.int64))
        # print(i, df[i].dtype)
        assert is_numeric_dtype(df[i])
    # df.to_csv("adult_no_missing.csv", index=False)
    # import ipdb; ipdb.set_trace()
    # Change all alphanumeric features into numerical categories
    df.to_csv("adult_redone.csv", index=False)
    # df_normalized = df.drop('target', axis=1)
    # df_normalized = df_normalized.apply(lambda x: (x - np.min(x)) / (np.max(x) - np.min(x)))        # corrected to min, instead of mean
    # df.to_csv("german_redone.csv", index=False)
    # convert_numerical_to_categorical(df)
    # import ipdb; ipdb.set_trace()
    # df1 = pd.get_dummies(df, columns=['Credit-history','Purpose','Other-debtors','Property','Other-installment-plans','Housing'])
    # convert_numerical_to_categorical(df1)
    # col = df1.pop("target")
    # df1.insert(42, col.name, col)
    # df1.to_csv("german_onehot.csv", index=False)
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
        
