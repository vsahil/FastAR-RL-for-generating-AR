import os
from joblib import Parallel, delayed


def run_this(i):
    os.system(f'python -W ignore classifier_german.py {i}')


Parallel(n_jobs=80)(delayed(run_this)(i) for i in range(100))
