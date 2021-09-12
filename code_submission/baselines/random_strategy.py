import sys, sklearn, gym, random, os, time
import numpy as np
sys.path.append("../fastar/gym-midline/gym_midline/envs/")
import german_credit as GR
import adult_income as AI
import credit_default as CD
import cal_metrics
import classifier_dataset as classifier
random.seed(42)
import pandas as pd


name = sys.argv[1]
if name == "german":
    x = GR.GermanCredit01()
elif name == "adult":
    x = AI.AdultIncome01()
elif name == "default":
    x = CD.CreditDefault01()

st = time.time()
if isinstance(x.action_space, gym.spaces.discrete.Discrete):
    total_actions = x.action_space.n
else:
    raise NotImplementedError

episode = 0
cfs_found = []
successful_knn_distance = []
successful_paths = []

print(len(x.undesirable_x), "See")
final_cfs = []
episodes = len(x.undesirable_x)
# episodes = 3
while episode < episodes:
    os.environ['SEQ'] = f"{episode}"
    x.reset()
    probability_class1 = x.classifier.predict_proba(x.state.reshape(1, -1))[0][1]	    # find the probability of belonging to class 1 - 
    knn_distances = 0
    for act in range(200):
        action = random.randint(0, total_actions-1)     # because it is generates numbers equal to total_actions
        state, reward, done, info = x.step(action)
        assert (state == x.state).all()
        knn_distances += x.distance_to_closest_k_points(state)
        if done:
            cfs_found.append(True)
            successful_knn_distance.append(knn_distances / (act+1))
            successful_paths.append(act)
            final_cfs.append(state)
            print("Success:", episode, act+1, knn_distances)
            break
    else:
        final_cfs.append(state)     # we will have a state even if not successful
        cfs_found.append(False)
    episode += 1

final_cfs = np.array(final_cfs)
if episodes < 20:
    print(cfs_found)
print(f"Success rate: {sum(cfs_found) * 100 /len(x.undesirable_x)}")
print(np.mean(successful_paths))
print(np.mean(successful_knn_distance))
print(successful_paths)

time_taken = time.time() - st

if "german" == name:
    continuous_features = ['Months', 'Credit-amount', 'Insatllment-rate', 'Present-residence-since', 'age', 'Number-of-existing-credits', 'Number-of-people-being-lible']
    immutable_features = ['Personal-status', 'Number-of-people-being-lible', 'Foreign-worker', 'Purpose']
    non_decreasing_features = ['age', 'Job']
    correlated_features = []
    name_dataset = "german"

elif "adult" == name:
    continuous_features = ['age', 'fnlwgt', 'capitalgain', 'capitalloss', 'hoursperweek']
    immutable_features = ['Marital-status', 'Race', 'Native-country', 'Sex']
    non_decreasing_features = ['age', 'education']
    correlated_features = [('education', 'age', 0.054)]     # in normalized data the increase is 0.05
    name_dataset = "adult"

elif "default" == name:
    continuous_features = ['LIMIT_BAL', 'AGE', 'BILL_AMT1', 'BILL_AMT2', 'BILL_AMT3', 'BILL_AMT4', 'BILL_AMT5', 'BILL_AMT6', 'PAY_AMT1', 'PAY_AMT2', 'PAY_AMT3', 'PAY_AMT4', 'PAY_AMT5', 'PAY_AMT6']
    immutable_features = ['sex', 'MARRIAGE']
    non_decreasing_features = ['AGE', 'EDUCATION']
    correlated_features = [('EDUCATION', 'AGE', 0.027)]     # in normalized data the increase is 0.027
    name_dataset = "default"

normalized_mads = {}
dataset = x.scaler.transform(x.dataset)
dataset = pd.DataFrame(dataset, columns=x.dataset.columns.tolist())
for feature in continuous_features:
    normalized_mads[feature] = np.median(abs(dataset[feature].values - np.median(dataset[feature].values)))
final_cfs = pd.DataFrame(final_cfs, columns=x.dataset.columns.tolist())
find_cfs_points = x.scaler.transform(x.undesirable_x[:episodes])
find_cfs_points = pd.DataFrame(find_cfs_points, columns=x.dataset.columns.tolist())
cal_metrics.calculate_metrics("random-" + name, final_cfs, cfs_found, find_cfs_points, x.classifier, x.dataset,
    continuous_features, normalized_mads, 
    immutable_features, non_decreasing_features, correlated_features, x.scaler, time_taken, save=False)
