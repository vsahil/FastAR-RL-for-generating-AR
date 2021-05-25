import sys, os, time
sys.path.append("/scratch/vsahil/RL-for-Counterfactuals/paper_expts/")
import dice_ml
from dice_ml.utils import helpers # helper functions
import pandas as pd
import numpy as np
import cal_metrics


# import ipdb; ipdb.set_trace()
# dataset_dice = helpers.load_adult_income_dataset()
# d = dice_ml.Data(dataframe=dataset_dice, continuous_features=['age', 'hours_per_week'], outcome_name='income')
import classifier_german as classifier
dataset_name = "german"      # "adult"
if dataset_name == "adult":
    file1 = "/scratch/vsahil/RL-for-Counterfactuals/paper_expts/adult_redone.csv"
    model, dataset, scaler, X_test, X_train = classifier.train_model_adult(file=file1, parameter=1, drop=False)
    continuous_features = ['age', 'fnlwgt', 'capitalgain', 'capitalloss', 'hoursperweek']
    immutable_features = ['marital-status', 'race', 'native-country', 'sex']
    non_decreasing_features = ['age', 'education']
    correlated_features = [('education', 'age', 2)]     # With each increase in level of education, we increase the age by 2. 
elif dataset_name == "german":
    file1 = "/scratch/vsahil/RL-for-Counterfactuals/paper_expts/german_redone.csv"
    model, dataset, scaler, X_test, X_train = classifier.train_model_german(file=file1, parameter=1, drop=False)
    continuous_features = ['Months', 'Credit-amount', 'Insatllment-rate', 'Present-residence-since', 'age', 'Number-of-existing-credits', 'Number-of-people-being-lible']
    immutable_features = ['Personal-status', 'Number-of-people-being-lible', 'Foreign-worker', 'Purpose']
    non_decreasing_features = ['age', 'Job']
    correlated_features = []
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
    # np.save(f"undesirable_x_{dataset}.npy", undesirable_x)

find_cfs_points = pd.DataFrame(undesirable_x, columns=X_test.columns.tolist())

model.predict_probability = model.predict_proba
model.predict_proba = predict_proba_dice

model.orig_predict = model.predict
model.predict = new_predict

# Using sklearn backend
m = dice_ml.Model(model=model, backend="sklearn")
# Using method=random for generating CFs
method = str(sys.argv[1])
# method = "kdtree"
# method = "genetic"
# method = "random"

exp = dice_ml.Dice(d, m, method=method)

start = time.time()
# immutable_features = []

# immutable_features = [i.lower() for i in immutable_features]
features_to_vary = [f for f in X_test.columns.tolist() if f not in immutable_features]
# import ipdb; ipdb.set_trace()
assert len(features_to_vary) == len(X_test.columns.tolist()) - len(immutable_features)

num_datapoints = find_cfs_points.shape[0]
# num_datapoints = 5

e1, cfs_found = exp.generate_counterfactuals(find_cfs_points[:num_datapoints], total_CFs=1, 
        desired_class="opposite", 
        features_to_vary=features_to_vary)

# import ipdb; ipdb.set_trace()
print(f"{method} : No. of found cfes: {sum(cfs_found)} Numdatapoints: {num_datapoints} Time taken: {time.time() - start}")
time_taken = time.time() - start
save = False
if save:
    with open("results.txt", "a") as f:
        print(f"{method + dataset_name}: {sum(cfs_found)} : {num_datapoints} : {time.time() - start}", file=f)

cal_metrics.calculate_metrics(method + dataset_name, e1, cfs_found, find_cfs_points, model, 
            dataset.drop(columns=['target']), continuous_features, d.get_mads(), 
            immutable_features, non_decreasing_features, correlated_features, scaler, time_taken, save=save)
