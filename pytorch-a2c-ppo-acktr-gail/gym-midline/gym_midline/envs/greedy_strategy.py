import sys, sklearn, gym, random, os, time
import numpy as np
import german_credit as GR
import adult_income as AI
import credit_default as CD
sys.path.append("/scratch/vsahil/RL-for-Counterfactuals/paper_expts/")
sys.path.append("../../../")
import cal_metrics
import classifier_german as classifier
import copy
random.seed(42)
import pandas as pd


name = "german"
if name == "german":
    x = GR.GermanCredit01()
elif name == "adult":
    x = AI.AdultIncome01()
elif name == "default":
    x = CD.CreditDefault01()

# import ipdb; ipdb.set_trace()
st = time.time()
if isinstance(x.action_space, gym.spaces.discrete.Discrete):
    total_actions = x.action_space.n
    # print(x.state)
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
    # print(f"Episode: {episode}")
    os.environ['SEQ'] = f"{episode}"
    x.reset()
    probability_class1 = x.classifier.predict_proba(x.state.reshape(1, -1))[0][1]	    # find the probability of belonging to class 1 - 
    # print(probability_class1, episode)
    # assert (probability_class1 < 0.5)
    knn_distances = 0
    # import ipdb; ipdb.set_trace()
    for act in range(200):
        old_state = copy.deepcopy(x.state)
        old_state_ = copy.deepcopy(x.state)
        # print(act, "see steps")
        max_reward = -1000
        chosen_action = None
        for action in range(total_actions):
            # print(act, action, total_actions, "check")
            state, reward, done, info = x.step(action)
            x.state = copy.deepcopy(old_state)
            if reward > max_reward:
                max_reward = reward
                chosen_action = action
            # try:
            #     assert (old_state == old_state_).all()
            # except:
            #     print(old_state - old_state_)
            #     print(old_state - x.state)
            #     print(x.state - old_state_)
            #     print("Done", act, action)
            #     assert False
        assert (x.state == old_state_).all()
        state, reward, done, info = x.step(chosen_action)
        knn_distances += x.distance_to_closest_k_points(state)
        # print(episode, act, chosen_action, done)
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

if "adult" == name:
    continuous_features = ['age', 'fnlwgt', 'capitalgain', 'capitalloss', 'hoursperweek']
    immutable_features = ['Marital-status', 'Race', 'Native-country', 'Sex']
    non_decreasing_features = ['age', 'education']
    correlated_features = [('education', 'age', 0.054)]     # in normalized data the increase is 0.05
    name_dataset = "adult"
elif "german" == name:
    numerical_features = x.numerical_features
    continuous_features = x.dataset.columns[numerical_features].tolist()
    immutable_features = ['Personal-status', 'Number-of-people-being-lible', 'Foreign-worker', 'Purpose']
    non_decreasing_features = ['age', 'Job']
    correlated_features = []
    name_dataset = "german"
# print(continuous_features)

normalized_mads = {}
dataset = x.scaler.transform(x.dataset)
dataset = pd.DataFrame(dataset, columns=x.dataset.columns.tolist())
for feature in continuous_features:
    normalized_mads[feature] = np.median(abs(dataset[feature].values - np.median(dataset[feature].values)))
final_cfs = pd.DataFrame(final_cfs, columns=x.dataset.columns.tolist())
find_cfs_points = x.scaler.transform(x.undesirable_x[:episodes])
find_cfs_points = pd.DataFrame(find_cfs_points, columns=x.dataset.columns.tolist())
# print(normalized_mads, "see")
# Adult: {'age': 0.2739726027397258, 'fnlwgt': 0.08197802435899859, 'capitalgain': 0.0, 'capitalloss': 0.0, 'hoursperweek': 0.061224489795918324}
cal_metrics.calculate_metrics("baseline", final_cfs, cfs_found, find_cfs_points, x.classifier, x.dataset,
    x.knn, continuous_features, normalized_mads, 
    immutable_features, non_decreasing_features, correlated_features, x.scaler, "greedy" + name, time_taken, save=False)
print("done")
