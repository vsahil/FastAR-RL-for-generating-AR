import sys, os, time
import dice_ml
from dice_ml.utils import helpers # helper functions
import pandas as pd
import numpy as np
import cal_metrics
import pickle
sys.path.append("../")
import classifier_dataset as classifier

dataset_name = sys.argv[1]

if dataset_name == "german":
    file1 = f"{os.path.dirname(os.path.realpath(__file__))}/../datasets/german_redone.csv"
    model, dataset, scaler, X_test, X_train = classifier.train_model_german(file=file1, parameter=1, drop=False)
    continuous_features = ['Months', 'Credit-amount', 'Insatllment-rate', 'Present-residence-since', 'age', 'Number-of-existing-credits', 'Number-of-people-being-lible']
    immutable_features = ['Personal-status', 'Number-of-people-being-lible', 'Foreign-worker', 'Purpose']
    non_decreasing_features = ['age', 'Job']
    correlated_features = []

elif dataset_name == "adult":
    file1 = f"{os.path.dirname(os.path.realpath(__file__))}/../datasets/adult_redone.csv"
    model, dataset, scaler, X_test, X_train = classifier.train_model_adult(file=file1, parameter=1, drop=False)
    continuous_features = ['age', 'fnlwgt', 'capitalgain', 'capitalloss', 'hoursperweek']
    immutable_features = ['marital-status', 'race', 'native-country', 'sex']
    non_decreasing_features = ['age', 'education']
    correlated_features = [('education', 'age', 2)]     # With each increase in level of education, we increase the age by 2. 

elif dataset_name == "default":
    file1 = f"{os.path.dirname(os.path.realpath(__file__))}/../datasets/default_redone.csv"
    model, dataset, scaler, X_test, X_train = classifier.train_model_default(file=file1, parameter=1, drop=False)
    continuous_features = ['LIMIT_BAL', 'AGE', 'BILL_AMT1', 'BILL_AMT2', 'BILL_AMT3', 'BILL_AMT4', 'BILL_AMT5', 'BILL_AMT6', 'PAY_AMT1', 'PAY_AMT2', 'PAY_AMT3', 'PAY_AMT4', 'PAY_AMT5', 'PAY_AMT6']
    immutable_features = ['sex', 'MARRIAGE']
    non_decreasing_features = ['AGE', 'EDUCATION']
    correlated_features = [('EDUCATION', 'AGE', 0.8)]   # the increase in unnormalized form is much higher. 

# X_train_ = scaler.transform(X_train)
# X_test_ = scaler.transform(X_test)

d = dice_ml.Data(dataframe=dataset, continuous_features=continuous_features, outcome_name='target')


def predict_proba_dice(input_instance):
    if isinstance(input_instance, pd.DataFrame):
        scaled_input = scaler.transform(input_instance)
        pr = model.predict_probability(scaled_input)
        # print(input_instance, scaled_input, pr)
        return pr
    else:
        print(input_instance, type(input_instance), "see")
        raise NotImplementedError


def new_predict(input_instance):
    pr = model.predict_proba(input_instance)
    classes = np.argmax(pr, axis=1)
    assert classes.shape[0] == input_instance.shape[0]
    return classes


try:
    undesirable_x = np.load(f"undesirable_x_{dataset_name}.npy")
    print("Found")
except:
    assert not os.path.exists(f"undesirable_x_{dataset_name}.npy")
    print("Not Found")
    undesirable_x = []
    for no, i in enumerate(X_test.to_numpy()):
        if classifier.predict_single(i, scaler, model) == 0:
            undesirable_x.append(tuple(i))
    undesirable_x = np.array(undesirable_x)

find_cfs_points = pd.DataFrame(undesirable_x, columns=X_test.columns.tolist())

model.predict_probability = model.predict_proba
model.predict_proba = predict_proba_dice

model.orig_predict = model.predict
model.predict = new_predict

# Using sklearn backend
m = dice_ml.Model(model=model, backend="sklearn")
# Using method=random for generating CFs
method = sys.argv[2]
# method = "kdtree"
# method = "genetic"
# method = "random"

exp = dice_ml.Dice(d, m, method=method)

start = time.time()
features_to_vary = [f for f in X_test.columns.tolist() if f not in immutable_features]
assert len(features_to_vary) == len(X_test.columns.tolist()) - len(immutable_features)

num_datapoints = find_cfs_points.shape[0]
# num_datapoints = 3
save_file = f'saved_files/{dataset_name}_explanations_{num_datapoints}.pkl'
if os.path.exists(save_file):
    print("loading saved results")
    with open(save_file, 'rb') as input:
        e1 = pickle.load(input)
    cfs_found = np.load(f'saved_files/{dataset_name}_cfs_found_{num_datapoints}.npy')

else:
    e1, cfs_found = exp.generate_counterfactuals(find_cfs_points[:num_datapoints], total_CFs=1, 
            desired_class="opposite", 
            features_to_vary=features_to_vary)

    # with open(save_file, 'wb') as output:
    #     pickle.dump(e1, output, pickle.HIGHEST_PROTOCOL)
    # np.save(f'saved_files/{dataset_name}_cfs_found_{num_datapoints}.npy', np.array(cfs_found))

save = False  # No need to save each independent evaluation metric as we are saving explanations. 

print(f"{method} : No. of found cfes: {sum(cfs_found)} Numdatapoints: {num_datapoints} Time taken: {time.time() - start}")
time_taken = time.time() - start
if save:
    with open("results.txt", "a") as f:
        print(f"{method + dataset_name}: {sum(cfs_found)} : {num_datapoints} : {time_taken}", file=f)

if num_datapoints > 5:
    # We need upper here so that "random" does not match the "random baseline" as they are handled differently by the cal_metrics file. 
    cal_metrics.calculate_metrics("dice-" + method.upper() + "-" + dataset_name, e1, cfs_found, find_cfs_points, model, 
            dataset.drop(columns=['target']), continuous_features, d.get_mads(), 
            immutable_features, non_decreasing_features, correlated_features, scaler, time_taken, save=save)

