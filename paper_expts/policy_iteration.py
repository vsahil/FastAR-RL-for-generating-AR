# Code partially taken from: https://medium.com/@annishared/searching-for-optimal-policies-in-python-an-intro-to-optimization-7182d6fe4dba

import numpy as np 
import pandas as pd 
import sys, os, copy
from sklearn.neural_network import MLPClassifier
# from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
import random, time
import itertools
from sklearn.neighbors import NearestNeighbors
import classifier


class environment:
	def __init__(self, n_states, n_actions, clf, dataset):	#, X_train=None, closest_points=None, dist_lambda=None):
		self.nS = n_states
		self.nA = n_actions
		self.classifier = clf
		self.states = {}
		self.states_reverse = {}
		self.state_count = 0
		self.action_count = 0
		self.dataset = dataset
		# self.P = [[0 for i in range(self.nA)] for j in range(self.nS)]	# this is the transition probabilities initialized. 
		# print("starting P")
		# a = time.time()
		# self.P = np.zeros((self.nS, self.nA))
		# print("created P", b - a)
		# b = time.time()
		self.state_sequence()
		# print("created dict", time.time() - b)
		# self.action_map = {'a1': 0, 'a2': 1, 'b1': 2, 'b2': 3, 'c1': 4} #, 'c2': 5}
		# self.reverse_action_map = {0: 'a1', 1: 'a2', 2: 'b1', 3: 'b2', 4: 'c1'} #, 5: 'c2'}

		
	def transition_function(self, state, action):
		# import ipdb; ipdb.set_trace()
		# present_state = list(list(self.states.keys())[list(self.states.values()).index(state)])
		present_state = list(self.states_reverse[state])
		feature_changing = action // 2	# this is the feature that is changing
		decrease = bool(action % 2)
		action_ = -1 if decrease else 1
		next_state = copy.deepcopy(present_state)
		next_state[feature_changing] = present_state[feature_changing] + action_
		values = sorted(self.dataset.iloc[:, feature_changing].unique())		# acces column feature changing
		if decrease:
			if present_state[feature_changing] > values[0]:
				try:
					next_state_ = self.states[tuple(next_state)]
				except:
					import ipdb; ipdb.set_trace()
				reward = self.model(next_state) * 10 - 1		# constant cost for each action		# - self.dist_lambda * self.distance_to_closest_k_points(a+1, b, c)
				return [(1.0, next_state_, reward, None)]
			else:
				return [(1.0, state, -10, None)]		# from the input itself
		else:
			if present_state[feature_changing] < values[-1]:
				try:
					next_state_ = self.states[tuple(next_state)]
				except:
					import ipdb; ipdb.set_trace()
				reward = self.model(next_state) * 10 - 1
				return [(1.0, next_state_, reward, None)]
			else:
				return [(1.0, state, -10, None)]
		

	def state_sequence(self):
		x = [dataset[i].unique() for i in self.dataset.columns]		# for all possible values in the dataset 
		for sts in list(itertools.product(*x)):
			self.states[sts] = self.state_count
			self.states_reverse[self.state_count] = sts
			self.state_count += 1

	
	def model(self, state):
		# import ipdb; ipdb.set_trace()
		# print("hello")
		return self.classifier.predict_proba(np.array([state]).reshape(1,-1))[0][1]		# find the probability of belonging to class 1 - 


	# def distance_to_closest_k_points(self, x, y, z):
	# 	nearest_dist, nearest_points = self.knn.kneighbors(np.array([y,z]).reshape(1,-1), self.no_points, return_distance=True)		# we will take the 5 closest points. We don't need 6 here because the input points are not training pts.
	# 	# quantity = np.mean(nearest_dist) / self.no_points
	# 	quantity = np.mean(nearest_dist)		# Now we have that lambda, so no need to divide by self.no_points 
	# 	# print((x,y,z), quantity)
	# 	return quantity


def policy_eval(policy, env, discount_factor=1.0, theta=0.00001, max_iterations=1000):
    # Initialize the value function
	V = np.zeros(env.nS)
    # While our value function is worse than the threshold theta
	iters = 0
	# while iters < max_iterations:
	# import ipdb; ipdb.set_trace()
	while True:
        # Keep track of the update done in value function
		delta = 0
        # For each state, look ahead one step at each possible action and next state
		for s in range(env.nS):
			v = 0
			# The possible next actions, policy[s]:[a, action_prob]
			for a, action_prob in enumerate(policy[s]): 
				# For each action, look at the possible next states, 
				# for prob, next_state, reward, done in env.P[s][a]: # state transition P[s][a] == [(prob, nextstate, reward, done), ...]
				for prob, next_state, reward, done in env.transition_function(s, a):
					# Calculate the expected value function
					v += action_prob * prob * (reward + discount_factor * V[next_state]) # P[s, a, s']*(R(s,a,s')+γV[s'])
					# How much our value function changed across any states .  
			# import ipdb; ipdb.set_trace()
			delta = max(delta, np.abs(v - V[s]))
			V[s] = v
		# Stop evaluating once our value function update is below a threshold
		print(iters, "done", delta)
		if delta < theta:
			break
		iters += 1
		
	return np.array(V)


def policy_improvement(env, policy_eval_fn=policy_eval, discount_factor=1.0):
    # Initialize a policy arbitarily
    policy = np.ones([env.nS, env.nA]) / env.nA			# initial same probability of taking all actions
    increment = 1.0 / env.nA 		# For instance, 0.2 for 5 states
    decrease = increment / (env.nA - 1)
    
    while True:
        # Compute the Value Function for the current policy
        a = time.time()
        print("starting")
        V = policy_eval_fn(policy, env, discount_factor)
        print("done", time.time() - a)

        # Will be set to false if we update the policy
        policy_stable = True

        # Improve the policy at each state
        for s in range(env.nS):
            # The best action we would take under the currect policy
            chosen_a = np.argmax(policy[s])
            # Find the best action by one-step lookahead
            action_values = np.zeros(env.nA)
            for a in range(env.nA):
                for prob, next_state, reward, done in env.P[s][a]:
                    action_values[a] += prob * (reward + discount_factor * V[next_state])
            best_a = np.argmax(action_values)		# what if instead of chosing the best action, I increment the probability of the best action by some fixed delta. 

            # Greedily (max in the above line) update the policy
            if chosen_a != best_a:
            	policy_stable = False
            
            # policy[s] = [policy[s][i] + increment if i == best_a else policy[s][i] - decrease for i in range(len(policy[s]))]
            # if min(policy[s]) > 1e-6:
            #     assert(int(sum(policy[s])) == 1)
            # else:
            #     policy[s] = np.eye(env.nA)[best_a]		# so this given the one-hot vector of policy for each state
            policy[s] = np.eye(env.nA)[best_a]
            # print("hello", int(sum(policy[s])), policy[s])
        # Until we've found an optimal policy. Return it
        if policy_stable:
            return policy, V


def create_synthetic_data(file):
	def sigmoid(x):
		return 1 / (1 + np.exp(-x))

	def age_finder(educational):
		for i, j in enumerate(educational):		# not able to find a vectorized version
			if j == 0:
				educational[i] = random.choice([0, 1, 2])
			elif j == 1:
				educational[i] = random.choice([1, 2])
			elif j == 2:
				educational[i] = random.choice([2, 3])
			elif j == 3:
				educational[i] = random.choice([2, 3])
			elif j == 4:
				educational[i] = random.choice([3, 4])
		
		return educational
	
	# import ipdb; ipdb.set_trace()
	# Now generate data where features have meanings: b is educational qualification and c is age. So there is a correlation between them. Feature a is still random. 
	graph_nodes_count = 4
	a = np.random.randint(0, 5, 100)
	b = np.random.randint(0, 5, 100)
	# c = np.random.randint(0, 5, 100)
	# now age will be correlated with educational qualification. 
	c = age_finder(copy.deepcopy(b))
	# for x,y,z, in zip(a,b,c):
	# 	print( ((x * x) - (y * z) ), sigmoid( (x * x) - (y * z)) )
	y = sigmoid( (a * a) - (b * c))		# changed the function, now we have almost a balanced dataset. 

	graph_data = np.zeros( (a.shape[0], graph_nodes_count), dtype=np.int64  )
	graph_data[:, 0] = a
	graph_data[:, 1] = b
	graph_data[:, 2] = c
	# import ipdb; ipdb.set_trace()
	for i in range(y.shape[0]):
		if y[i] > 0.5:
			graph_data[i, 3] = 1
		else:
			graph_data[i, 3] = 0        
	graph_data = pd.DataFrame(graph_data, columns=['a', 'b', 'c', 'y'] )
	graph_data.to_csv(file, index=False)


def train_model(file):
	# import ipdb; ipdb.set_trace()
	# X, y = make_classification(n_samples=100, random_state=1)
	total_dataset = pd.read_csv(file)
	Y = total_dataset['y']
	total_dataset = total_dataset.drop(columns=['y'])
	X_train, X_test, y_train, y_test = train_test_split(total_dataset, Y, stratify=Y, random_state=1)
	clf = MLPClassifier(random_state=1, max_iter=300).fit(X_train, y_train)
	# clf.predict_proba(X_test[:1])
	return clf, X_train
	# clf.predict(X_test[:5, :])

	clf.score(X_test, y_test)


def use_policy(policy, classifier, env, file):
	def take_action(a, b, c, action):
		if action == "a1" and a <= 3:
			return (a+1, b, c)
		elif action == "a2" and a >= 1:
			return (a-1, b, c)
		elif action == "b1" and b <= 3:
			return (a, b+1, c)
		elif action == "b2" and b >= 1:
			raise NotImplementedError
			return (a, b-1, c)
		elif action == "c1" and c <= 3:
			return (a, b, c+1)
		elif action == "c2" and c >= 1:
			raise NotImplementedError 		# c2 is not more a valid action
			return (a, b, c-1)
	
	
	def return_counterfactual(original_individual, transit):
		cost = 0
		individual = copy.deepcopy(original_individual)
		number = env.state_sequence(*individual)
		maxtry = 30
		attempt_no = 0
		while attempt_no < maxtry:
			action_ = np.where(policy[number] == 1)[0]
			assert len(action_) == 1
			action = env.reverse_action_map[action_[0]]
			new_pt = np.array(take_action(*individual, action))
			cost += 1
			attempt_no += 1
			if classifier.predict(new_pt.reshape(1, -1)) == 1:
				transit += 1
				print(original_individual, f"successful: {new_pt}",  cost)
				# total_cost += cost
				return transit, cost, env.distance_to_closest_k_points(*new_pt)		# the last term gives the Knn distance from k nearest points
			else:
				number = env.state_sequence(*new_pt)
				if (new_pt == individual).all():
					print("unsuccessful: ", original_individual)
					return transit, cost, env.distance_to_closest_k_points(*new_pt)
				individual = new_pt
		else:
			print("unsuccessful: ", original_individual)
			return transit, cost, env.distance_to_closest_k_points(*new_pt)


	total_dataset = pd.read_csv(file)
	Y = total_dataset['y']
	total_dataset = total_dataset.drop(columns=['y'])
	X_train, X_test, y_train, y_test = train_test_split(total_dataset, Y, stratify=Y, random_state=1)
	undesirable_x = X_test[y_test == 0].to_numpy()
	successful_transitions = 0
	total_cost = 0
	knn_dist = 0
	for no_, individual in enumerate(undesirable_x):
		transit, cost, single_knn_dist = return_counterfactual(individual, successful_transitions)
		if transit > successful_transitions:
			successful_transitions = transit
			total_cost += cost
			knn_dist += single_knn_dist

	try:
		avg_cost = total_cost / successful_transitions
		print(successful_transitions, len(undesirable_x), avg_cost, knn_dist)			
	except:		# due to zero division error 
		pass
	# print("see")
	success_rate = successful_transitions / len(undesirable_x)
	return success_rate, avg_cost, knn_dist


if __name__ == "__main__":
	clf, dataset, scaler = classifier.train_model(1)	
	n_actions = 2 * (len(dataset.columns) - 1)		# remove the target from set of features
	n_states = np.prod([len(dataset[i].unique()) for i in dataset.columns])		# no of discrete states is the number of unique values for each feature. 
	gamma = 0.99
	full_result_table = []
	print(n_states)
    # import ipdb; ipdb.set_trace()
	# closest_points = 10
	# seq_gamma = [0.1, 0.5, 0.9]
	# seq_closest_points = [1, 2, 5, 10]
	# dist_lambdas = [10, 5, 1, 0.5, 0.1, 0.05, 0.01]
	# dist_lambdas.reverse()
	# dist_lambda = 3 		# largest safe value. 
	experiment = False

	if experiment:
		full_result = {}
		for gamma in seq_gamma:
			for closest_points in seq_closest_points:
				success_for_gamma_pt = []
				cost_for_gamma_pt = []
				knn_dist_for_gamma_pt = []
				for dist_lambda in dist_lambdas:
					env = environment(n_states, n_actions, clf, X_train=X_train, closest_points=closest_points, dist_lambda=dist_lambda)
					# import ipdb; ipdb.set_trace()
					final_policy, V = policy_improvement(env, discount_factor=gamma)
					percentage_success, avg_cost, knn_dist = use_policy(final_policy, clf, env, file)
					success_for_gamma_pt.append(percentage_success)
					cost_for_gamma_pt.append(avg_cost)
					knn_dist_for_gamma_pt.append(knn_dist)
				# with open("output.py", "a") as f:
				# 	print(f'\nresults_{gamma}_{closest_points} = ', file=f)
				# 	print(success_for_gamma_pt, file=f)
				# 	print(cost_for_gamma_pt, file=f)
				# 	print(knn_dist_for_gamma_pt, file=f)
				full_result[f'{gamma}_{closest_points}'] = [success_for_gamma_pt, cost_for_gamma_pt, knn_dist_for_gamma_pt]
				# print(gamma, closest_points, success_for_gamma_pt, cost_for_gamma_pt, knn_dist_for_gamma_pt, "done")
			# full_result_table.append(success_for_gamma)
		with open("output.py", "a") as f:
			print(full_result, file=f)
		# exit(0)
		# headers = ["Closest Points", "Gamma = .1", "Gamma = .2", "Gamma = .5", "Gamma = .8", "Gamma = .9"]
		# table = np.array([seq_closest_points, np.around(np.array(full_result_table[0]), decimals=3), np.around(np.array(full_result_table[1]), decimals=3), np.around(np.array(full_result_table[2]), decimals=3), np.around(np.array(full_result_table[3]), decimals=3), np.around(np.array(full_result_table[4]), decimals=3)]).T
		# create_pdf_from_table('success_causal_dist_k_sweep', table, headers, mode="a")
	
	else:
		env = environment(n_states, n_actions, clf, dataset) #, X_train=X_train, closest_points=closest_points, dist_lambda=dist_lambda)
		# import ipdb; ipdb.set_trace()
		final_policy, V = policy_improvement(env, discount_factor=gamma)
		# percentage_success, avg_cost, knn_dist = use_policy(final_policy, clf, env, file)
	
	print("DONE")



# Experiments with stochastic policy while iterating
# time - 14s. No change in successful_transitions
# Iterations with gamma = 0.9: 90, 78, 71, 65, 61, 62, 62, 62, 63, 63, 63, 63: more iterations. Completely useless exercise. 
# Experiments with argmax policy: 
# time - 13.5s. so this is fast as well, no need of using this. But this is not the same as number of iteration. 
# Iterations with gamma = 0.9: 90, 61, 62, 62, 62, 63, 63, 63
