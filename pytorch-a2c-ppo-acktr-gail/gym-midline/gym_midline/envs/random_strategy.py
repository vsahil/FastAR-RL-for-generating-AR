import sys, sklearn, gym, random, os
import numpy as np
import german_credit as GR
import adult_income as AI
import credit_default as CD
sys.path.append("/scratch/vsahil/RL-for-Counterfactuals/paper_expts/")
import classifier_german as classifier
random.seed(42)

# x = GR.GermanCredit01()
# x = AI.AdultIncome01()
x = CD.CreditDefault01()
# import ipdb; ipdb.set_trace()
if isinstance(x.action_space, gym.spaces.discrete.Discrete):
    total_actions = x.action_space.n
    # print(x.state)
else:
    raise NotImplementedError

episode = 0
cf_reached = 0
successful_knn_distance = []
successful_paths = []

# specific = 2
# episode = specific
# import ipdb; ipdb.set_trace()
print(len(x.undesirable_x), "See")
while episode < len(x.undesirable_x):
    # print(f"Episode: {episode}")
    os.environ['SEQ'] = f"{episode}"
    x.reset()
    probability_class1 = x.classifier.predict_proba(x.state.reshape(1, -1))[0][1]	    # find the probability of belonging to class 1 - 
    assert (probability_class1 < 0.5)
    knn_distances = 0
    for act in range(200):
        action = random.randint(0, total_actions-1)     # because it is generates numbers equal to total_actions
        state, reward, done, info = x.step(action)
        assert (state == x.state).all()
        # print(episode, act, done)
        knn_distances += x.distance_to_closest_k_points(state)
        if done:
            cf_reached += 1
            successful_knn_distance.append(knn_distances / (act+1))
            successful_paths.append(act)
            print("Success:", episode, act+1, knn_distances)
            break
    episode += 1

print(cf_reached)
print(f"Success rate: {cf_reached/len(x.undesirable_x)}")
print(np.mean(successful_paths))
print(np.mean(successful_knn_distance))
print(successful_paths)
