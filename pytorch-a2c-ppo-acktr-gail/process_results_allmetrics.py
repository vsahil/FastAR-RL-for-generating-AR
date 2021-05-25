import numpy as np 
import pandas as pd

# df = pd.read_csv("all_metrics_our.csv")
df = pd.read_csv("all_metrics_ourgerman.csv")
print(df['validity'].max())
df = df[df['validity'] == df['validity'].max()]
# print(df)
df = df[df['proximity_cat'] == df['proximity_cat'].min()]       # bad validity for german
# print(df)
df = df[df['sparsity'] == df['sparsity'].min()]     # bad validity for german
# print(df)
df = df[df['causality'] == df['causality'].max()]      # everything
# print(df)
df['causality'] *= 100.0
df = df[df['manifold'] == df['manifold'].min()]
# print(df.nsmallest(10, 'manifold', keep='last'))        # even the 10 smallest manifold have very bad validity for german
# print(df)
df['manifold'] = df['manifold'].round(2)
df = df[df['proximity_cont'] == df['proximity_cont'].min()]
df['proximity_cont'] = df['proximity_cont'].round(2)
df = df[df['time'] == df['time'].min()]
df['time'] = df['time'].round(1)
df = df[['setting', 'validity', 'proximity_cont', 'proximity_cat', 'sparsity', 'manifold', 'causality', 'time']]
print(df)

'''

Adult Income dataset:
Our                   setting  validity  proximity_cont  proximity_cat  sparsity  manifold  causality   time
1  01_0.99_256_0.0001_0.2     100.0            0.04            0.0       1.0      0.18      100.0  110.1
         validity  proximity_cont  proximity_cat  sparsity  manifold  causality      time
random    100.000            0.82           0.04      1.64      1.24       90.0   1615.7
genetic    89.528            0.71           0.27      4.43      0.46       23.0  24807.0
kdtree      0.000            0.00           0.00      0.00      0.00        0.0   4235.7
greedy,97.66219394107068,0.03615034206949786,0.01951133144475921,1.1824362606232295,0.17441515677262523,0.9490084985835694,1919.578363418579
random,80.91022271406834,0.5608948141918065,0.7647033680971106,10.073687809882031,0.9976572139533345,0.2926996067703881,1838.938199520111
'''

'''
German Credit dataset
Our            setting   validity  proximity_cont  proximity_cat  sparsity  manifold  causality  time
3  01_0.99_256_0.001_0.2  97.276265             0.1       0.063385      1.22      0.72      100.0  17.9

'''