import pandas as pd
import numpy as np


def sort(file):
    df = pd.read_csv(file, comment = '#')
    df = df.sort_values(by=['Lambda'])
    # print(df)
    df.to_csv(file, index=False, mode='a')


def avg_values(file):
    df = pd.read_csv(file, comment = '#', dtype=np.float64)
    df_ = df[df['Lambda'] == 100]
    # import ipdb; ipdb.set_trace()
    total_pts = 222
    print(df_)
    print(round(df_["Correct"].mean()/222 * 100, 1), round(df_["KNN"].mean(), 2), round(df_["Path"].mean(), 1))


if __name__ == "__main__":
    # file = "correct_german_onehot_sampletrain.csv"
    file = "correct_german_onehot_contiaction_sampletrain.csv"
    # sort(file)
    avg_values(file)
