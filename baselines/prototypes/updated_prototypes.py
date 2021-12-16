# # Counterfactual explanations with ordinally encoded categorical variables

# This example notebook illustrates how to obtain [counterfactual explanations](https://docs.seldon.io/projects/alibi/en/latest/methods/CFProto.html) for instances with a mixture of ordinally encoded categorical and numerical variables. A more elaborate notebook highlighting additional functionality can be found [here](./cfproto_cat_adult_ohe.ipynb). We generate counterfactuals for instances in the *adult* dataset where we predict whether a person's income is above or below $50k.

# In[31]:


import tensorflow as tf
tf.get_logger().setLevel(40) # suppress deprecation messages
tf.compat.v1.disable_v2_behavior() # disable TF2 behaviour as alibi code still relies on TF1 constructs 
from tensorflow.keras.layers import Dense, Input, Embedding, Concatenate, Reshape, Dropout, Lambda
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.utils import to_categorical

import numpy as np
import os
from sklearn.preprocessing import OneHotEncoder
import time
from alibi.datasets import fetch_adult
from alibi.explainers import CounterfactualProto

print('TF version: ', tf.__version__)
print('Eager execution enabled: ', tf.executing_eagerly()) # False

import sys
sys.path.append("../../")
import classifier_dataset as classifier
dataset_name = sys.argv[1]

if dataset_name == "german":
    dataset, scaler, X_test, X_train, y_train, y_test = classifier.train_model_german(parameter=3)
    continuous_features = ['Months', 'Credit-amount', 'Insatllment-rate', 'Present-residence-since', 'age', 'Number-of-existing-credits', 'Number-of-people-being-lible']
    immutable_features = ['Personal-status', 'Number-of-people-being-lible', 'Foreign-worker', 'Purpose']
    non_decreasing_features = ['age', 'Job']
    correlated_features = []
    epochs = 100

elif dataset_name == "adult":
    dataset, scaler, X_test, X_train, y_train, y_test = classifier.train_model_adult(parameter=3)
    continuous_features = ['age', 'fnlwgt', 'capitalgain', 'capitalloss', 'hoursperweek']
    immutable_features = ['marital-status', 'race', 'native-country', 'sex']
    non_decreasing_features = ['age', 'education']
    correlated_features = [('education', 'age', 0.054)]     # With each increase in level of education, we increase the age by 2. 
    epochs = 25

elif dataset_name == "default":
    dataset, scaler, X_test, X_train, y_train, y_test = classifier.train_model_default(parameter=3)
    continuous_features = ['LIMIT_BAL', 'AGE', 'BILL_AMT1', 'BILL_AMT2', 'BILL_AMT3', 'BILL_AMT4', 'BILL_AMT5', 'BILL_AMT6', 'PAY_AMT1', 'PAY_AMT2', 'PAY_AMT3', 'PAY_AMT4', 'PAY_AMT5', 'PAY_AMT6']
    immutable_features = ['sex', 'MARRIAGE']
    non_decreasing_features = ['AGE', 'EDUCATION']
    correlated_features = [('EDUCATION', 'AGE', 0.027)]   # the increase in unnormalized form is much higher. 
    epochs = 10

X_train_ = scaler.transform(X_train)
X_test_ = scaler.transform(X_test)

def set_seed(s=0):
    np.random.seed(s)
    tf.random.set_seed(s)


def nn_ord():
    
    x_in = Input(shape=(13,))
    layers_in = []
    
    # # embedding layers
    # for i, (_, v) in enumerate(cat_vars_ord.items()):
    #     emb_in = Lambda(lambda x: x[:, i:i+1])(x_in)
    #     emb_dim = int(max(min(np.ceil(.5 * v), 50), 2))
    #     emb_layer = Embedding(input_dim=v+1, output_dim=emb_dim, input_length=1)(emb_in)
    #     emb_layer = Reshape(target_shape=(emb_dim,))(emb_layer)
    #     layers_in.append(emb_layer)
        
    # numerical layers
    num_in = Lambda(lambda x: x[:])(x_in)
    num_layer = Dense(13)(num_in)
    layers_in.append(num_layer)
    
    # combine
    x = Concatenate()(layers_in)
    x = Dense(60, activation='relu')(x)
    x = Dropout(.2)(x)
    x = Dense(60, activation='relu')(x)
    x = Dropout(.2)(x)
    x = Dense(60, activation='relu')(x)
    x = Dropout(.2)(x)
    x_out = Dense(2, activation='softmax')(x)
    
    nn = Model(inputs=x_in, outputs=x_out)
    nn.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    
    return nn


def our_model(input_dim):
    model = Sequential()
    model.add(Dense(5, input_dim=input_dim, activation='relu'))
    model.add(Dense(3, activation='relu'))
    # model.add(Dense(1, activation='sigmoid'))
    model.add(Dense(2, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


set_seed()
nn = our_model(input_dim=X_train_.shape[1])
nn.fit(X_train_, to_categorical(y_train), epochs=epochs, batch_size=200)

predictions = np.argmax(nn.predict(X_test_), axis=1)
test_accuracy = sum(predictions == y_test.to_numpy()) * 100.0 / len(y_test)
print("Test Accuracy: ", test_accuracy)

shape = (1,) + X_test_.shape[1:]
beta = .01
c_init = 1.
c_steps = 5
max_iterations = 500
rng = (-1., 1.)  # scale features between -1 and 1
rng_shape = (1,) + X_test_.shape[1:]
feature_range = ((np.ones(rng_shape) * rng[0]).astype(np.float32), 
                 (np.ones(rng_shape) * rng[1]).astype(np.float32))

set_seed()
# define predict function
predict_fn = lambda x: nn.predict(x)

cf = CounterfactualProto(predict_fn,
                         shape,
                         beta=beta,
                         cat_vars=None,     # cat_vars_ord,
                         max_iterations=max_iterations,
                         feature_range=feature_range,
                         c_init=c_init,
                         c_steps=c_steps,
                         eps=(.01, .01)  # perturbation size for numerical gradients
                        )

# Fit explainer. Please check the [documentation](https://docs.seldon.io/projects/alibi/en/latest/methods/CFProto.html) for more info about the optional arguments.
# import ipdb; ipdb.set_trace()

cf.fit(X_train_) #, d_type='abdm', disc_perc=[25, 50, 75]);
# Find CF for instances with original predictions 0
if dataset_name == "german":
    predictions_train = np.argmax(nn.predict(X_train_), axis=1)
    predictions = np.concatenate((predictions_train, predictions))
predictions_with_0 = np.where(predictions == 0)[0]

print("DATASET: ", dataset_name, predictions_with_0.shape)

start = time.time()

set_seed()
# As this approach is very expensive to run, I am running it on the first 100 datapoints only.
num_datapoints = 1000
cfs_found = []
final_cfs = []
save = True

for num, dts in enumerate(predictions_with_0[:num_datapoints]):
    explanation = cf.explain(X_test_[dts].reshape(1, -1), verbose=False)
    if explanation.cf is not None:
        X_cf_ord = explanation.cf['X']
        print("Original: ", X_test_[dts])
        print("counterfactual: ", X_cf_ord)
        final_cfs.append(X_cf_ord)
        cfs_found.append(1)
    else:
        print("No counterfactual found")
        final_cfs.append(X_test_[dts])      # this will be a dummy
        cfs_found.append(0)
    if num % 10 == 0:
        print("CF seq: ", num)

method = "CFproto-"
print(f"{method} : No. of found cfes: {sum(cfs_found)} Numdatapoints: {num_datapoints} Time taken: {time.time() - start}")
time_taken = time.time() - start

if save:
    with open("results.txt", "a") as f:
        print(f"{method + dataset_name}: {sum(cfs_found)} : {num_datapoints} : {time_taken}", file=f)

sys.path.append("../")
import cal_metrics
dataset = dataset.drop(columns=['target'])
normalized_mads = {}
dataset_ = scaler.transform(dataset)
import pandas as pd
dataset_ = pd.DataFrame(dataset_, columns=dataset.columns.tolist())
for feature in continuous_features:
    normalized_mads[feature] = np.median(abs(dataset_[feature].values - np.median(dataset_[feature].values)))
final_cfs = np.array(final_cfs)
final_cfs = final_cfs.squeeze()
final_cfs = pd.DataFrame(final_cfs, columns=dataset.columns.tolist())
undesirable_x = X_test_[predictions_with_0]
find_cfs_points = pd.DataFrame(undesirable_x[:num_datapoints], columns=dataset.columns.tolist())

cal_metrics.calculate_metrics(method + dataset_name, final_cfs, cfs_found, find_cfs_points[:num_datapoints], 
        nn, dataset, continuous_features, normalized_mads, 
        immutable_features, non_decreasing_features, correlated_features, scaler, time_taken, save=save)




# Helper function to more clearly describe explanations:

# def describe_instance(X, explanation, eps=1e-2):
#     target_names = ['<=50K', '>50K']
#     print('Original instance: {}  -- proba: {}'.format(target_names[explanation.orig_class],
#                                                        explanation.orig_proba[0]))
#     print('Counterfactual instance: {}  -- proba: {}'.format(target_names[explanation.cf['class']],
#                                                              explanation.cf['proba'][0]))
#     print('\nCounterfactual perturbations...')
#     print('\nCategorical:')
#     X_orig_ord = X
#     X_cf_ord = explanation.cf['X']
#     delta_cat = {}
#     for i, (_, v) in enumerate(category_map.items()):
#         cat_orig = v[int(X_orig_ord[0, i])]
#         cat_cf = v[int(X_cf_ord[0, i])]
#         if cat_orig != cat_cf:
#             delta_cat[feature_names[i]] = [cat_orig, cat_cf]
#     if delta_cat:
#         for k, v in delta_cat.items():
#             print('{}: {}  -->   {}'.format(k, v[0], v[1]))
    
#     print('\nNumerical:')
#     delta_num = X_cf_ord[0, -4:] - X_orig_ord[0, -4:]
#     n_keys = len(list(cat_vars_ord.keys()))
#     for i in range(delta_num.shape[0]):
#         if np.abs(delta_num[i]) > eps:
#             print('{}: {:.2f}  -->   {:.2f}'.format(feature_names[i+n_keys],
#                                             X_orig_ord[0,i+n_keys],
#                                             X_cf_ord[0,i+n_keys]))

# describe_instance(X, explanation)

