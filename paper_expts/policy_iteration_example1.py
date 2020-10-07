# Code partially taken from: https://medium.com/@annishared/searching-for-optimal-policies-in-python-an-intro-to-optimization-7182d6fe4dba
# Parallelization has made this code almost 30*30 = 900 times faster. 
# After all parallelization this is still not fast enough because I was calling a function. 
# Replaced it by a pre-computed array, so constant lookup is so much faster now, can make the states larger back. 
# In pre-computed array, parallelization made it slow. 


import numpy as np 
import pandas as pd 
import sys, os, copy
from sklearn.neural_network import MLPClassifier
# from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
import random, time
import itertools
from sklearn.neighbors import NearestNeighbors
from joblib import Parallel, delayed
import multiprocessing
import classifier


class environment:
	def __init__(self, n_states, n_actions, clf, discount_factor, scaler, dataset):	#, X_train=None, closest_points=None, dist_lambda=None):
		self.nS = n_states
		self.nA = n_actions
		self.classifier = clf
		self.states = {}
		self.states_reverse = {}
		self.state_count = 0
		self.action_count = 0
		self.no_neighbours = 1
		self.scaler = scaler
		self.dataset = dataset
		self.discount_factor = discount_factor
		# self.P = np.zeros((self.nS, self.nA))		# can't be numpy array as I will replace it with list
		self.P = [[0 for i in range(self.nA)] for j in range(self.nS)]
		self.state_sequence()
		self.transition_function_version2()
		
		self.knn = NearestNeighbors(n_neighbors=5, p=1)		# 1 would be self, L1 distance makes sense for categorical features
		self.knn.fit(self.dataset)
	

	def transition_function_version2(self):
		x = [self.dataset[i].unique() for i in self.dataset.columns]		# for all possible values in the dataset 
		for sts in list(itertools.product(*x)):
			for act in range(self.nA):
				sts_sequence = self.states[sts]
				feature_changing = act // 2		# this is the feature that is changing
				decrease = bool(act % 2)
				action_ = -1 if decrease else 1
				next_state = list(copy.deepcopy(sts))
				next_state[feature_changing] = sts[feature_changing] + action_
				values = sorted(self.dataset.iloc[:, feature_changing].unique())		# acces column feature changing

				if decrease:
					if sts[feature_changing] > values[0]:
						next_state_ = self.states[tuple(next_state)]
						reward = self.model(next_state) - 1		# constant cost for each action		# - self.dist_lambda * self.distance_to_closest_k_points(a+1, b, c)
						ret = [(1.0, next_state_, reward, None)]
					else:
						ret = [(1.0, sts_sequence, -10, None)]		# from the input itself
				else:
					if sts[feature_changing] < values[-1]:
						next_state_ = self.states[tuple(next_state)]
						reward = self.model(next_state) - 1
						ret = [(1.0, next_state_, reward, None)]
					else:
						ret = [(1.0, sts_sequence, -10, None)]
				self.P[sts_sequence][act] = ret
	

	def state_sequence(self):
		x = [self.dataset[i].unique() for i in self.dataset.columns]		# for all possible values in the dataset 
		for sts in list(itertools.product(*x)):
			self.states[sts] = self.state_count
			self.states_reverse[self.state_count] = sts
			self.state_count += 1
		assert self.state_count == self.nS

	
	def model(self, state):
		# import ipdb; ipdb.set_trace()
		# print("hello")
		# if classifier.predict_single(state, self.scaler, self.classifier) == 1:
		arr = np.array([state])
		arr = self.scaler.transform(arr.reshape(1, -1))
		probability_class1 = self.classifier.predict_proba(arr.reshape(1,-1))[0][1]	# find the probability of belonging to class 1 - 
		if probability_class1 >= 0.5:
			try:
				assert classifier.predict_single(state, self.scaler, self.classifier) == 1
			except:
				import ipdb; ipdb.set_trace()
			return 10		# if it is already in good state then very high reward, this should help us get 100% success rate hopefully
		return probability_class1		#, multiply by 2 to encourage going to positively classified states	- 100% sucesss and 1.75 cost
										# multiply by 1, - 100% sucesss and 1.75 cost
										# multiply by 1, - 100% sucesss and 1.75 cost


	def distance_to_closest_k_points(self, state):
		nearest_dist, nearest_points = self.knn.kneighbors(np.array([state]).reshape(1,-1), self.no_neighbours, return_distance=True)		# we will take the 5 closest points. We don't need 6 here because the input points are not training pts.
		# quantity = np.mean(nearest_dist) / self.no_points
		quantity = np.mean(nearest_dist)		# Now we have that lambda, so no need to divide by self.no_points 
		# print(quantity, nearest_dist)
		return quantity


def hepler_policy_eval(policy, V, s):
	global env
	v = 0
	delta = 0
	# The possible next actions, policy[s]:[a, action_prob]
	for a, action_prob in enumerate(policy[s]): 
		# For each action, look at the possible next states, 
		for prob, next_state, reward, done in env.P[s][a]: 	# state transition P[s][a] == [(prob, nextstate, reward, done), ...]
		# for prob, next_state, reward, done in env.transition_function(s, a):
			# Calculate the expected value function
			v += action_prob * prob * (reward + env.discount_factor * V[next_state]) # P[s, a, s']*(R(s,a,s')+γV[s'])
			# How much our value function changed across any states .  
	delta = max(delta, np.abs(v - V[s]))
	# V[s] = v
	# print(s, "done", v)
	return v, s, delta


def policy_eval(policy, theta=1e-5, max_iterations=1000):
    # Initialize the value function
	global env
	V = np.zeros(env.nS)
    # While our value function is worse than the threshold theta
	iters = 0
	# while iters < max_iterations:
	# ses = [s for s in range(env.nS)]
	while True:
        # Keep track of the update done in value function
		delta = 0
        # For each state, look ahead one step at each possible action and next state
		for s in range(env.nS):
			v = 0
			# The possible next actions, policy[s]:[a, action_prob]
			for a, action_prob in enumerate(policy[s]): 
				# For each action, look at the possible next states, 
				for prob, next_state, reward, done in env.P[s][a]: 	# state transition P[s][a] == [(prob, nextstate, reward, done), ...]
				# for prob, next_state, reward, done in env.transition_function(s, a):
					# Calculate the expected value function
					v += action_prob * prob * (reward + env.discount_factor * V[next_state]) # P[s, a, s']*(R(s,a,s')+γV[s'])
					# How much our value function changed across any states .  
			# import ipdb; ipdb.set_trace()
			delta = max(delta, np.abs(v - V[s]))
			V[s] = v
			# print(s, "done")
		# Stop evaluating once our value function update is below a threshold
		
		# st = time.time()
		# pool = multiprocessing.Pool(multiprocessing.cpu_count())
		# x = zip(itertools.repeat(policy), itertools.repeat(V), ses)
		# result = pool.starmap_async(hepler_policy_eval, x)
		# pool.close()
		
		# res = result.get()
		# # return v, s, delta
		# for i in res:
		# 	V[i[1]] = i[0]
		# delta = max([i[2] for i in res])
		# print(iters, "iters done", delta) 	#, time.time() - st)
		# import ipdb; ipdb.set_trace()
		if delta < theta:
			break
		iters += 1
		
	return np.array(V)


def helper_policy_improve(policy, V, s):
	global env
	policy_stable = True
	chosen_a = np.argmax(policy[s])
	# Find the best action by one-step lookahead
	action_values = np.zeros(env.nA)
	for a in range(env.nA):
		for prob, next_state, reward, done in env.P[s][a]:
		# for prob, next_state, reward, done in env.transition_function(s, a):	
			action_values[a] += prob * (reward + env.discount_factor * V[next_state])
	best_a = np.argmax(action_values)		# what if instead of chosing the best action, I increment the probability of the best action by some fixed delta. 

	# Greedily (max in the above line) update the policy
	if chosen_a != best_a:
		policy_stable = False
	
	return best_a, s, policy_stable
	# policy[s] = np.eye(env.nA)[best_a]


def policy_improvement(policy_eval_fn=policy_eval):
    # Initialize a policy arbitarily
	global env
	epoch = 0
	policy = np.ones([env.nS, env.nA]) / env.nA			# initial same probability of taking all actions
	increment = 1.0 / env.nA 		# For instance, 0.2 for 5 states
	decrease = increment / (env.nA - 1)
	# ses = [s for s in range(env.nS)]
	while True:
        # Compute the Value Function for the current policy
		# V = policy_eval_fn(policy=policy, env, discount_factor=env.discount_factor)
		V = policy_eval_fn(policy=policy)

		# Will be set to false if we update the policy
		policy_stable = True
		no_false = 0
        # Improve the policy at each state
		for s in range(env.nS):	# this loop also requires about 38s, down to 1.3s
            # The best action we would take under the currect policy
			chosen_a = np.argmax(policy[s])
			# Find the best action by one-step lookahead
			action_values = np.zeros(env.nA)
			for a in range(env.nA):
				for prob, next_state, reward, done in env.P[s][a]:
				# for prob, next_state, reward, done in env.transition_function(s, a):	
					action_values[a] += prob * (reward + env.discount_factor * V[next_state])
			best_a = np.argmax(action_values)		# what if instead of chosing the best action, I increment the probability of the best action by some fixed delta. 

            # Greedily (max in the above line) update the policy
			if chosen_a != best_a:
				no_false += 1
				policy_stable = False
            
			policy[s] = np.eye(env.nA)[best_a]
			
			# print(s, "in improvement")
		# pool = multiprocessing.Pool(multiprocessing.cpu_count())
		# result = pool.starmap_async(helper_policy_improve, zip(itertools.repeat(policy), itertools.repeat(V), ses))
		# pool.close()
		
		# res = result.get()
		# # return best_a, s, policy_stable
		# for i in res:
		# 	policy[i[1]] = np.eye(env.nA)[i[0]]
		# policy_stable_ = False if False in [i[2] for i in res] else True	# even if one False, make it False
		print(f"done policy improvemnt epoch:{epoch}", " and policy stable:", policy_stable, "no falses: ", no_false)
		epoch += 1
		# Until we've found an optimal policy. Return it
		if policy_stable:
			return policy, V


def use_policy(policy, V, X_test):
	global env
	
	def return_counterfactual(original_individual, transit):
		path_len = 0
		individual = copy.deepcopy(original_individual)
		number = env.states[individual]
		maxtry = 100
		path = [original_individual]
		while path_len < maxtry:
			action_ = np.where(policy[number] == 1)[0]
			assert len(action_) == 1
			# action = env.reverse_action_map[action_[0]]
			_, next_state, _, _ = env.P[number][action_[0]][0]	# need to take out of tuple
			new_pt = env.states_reverse[next_state]
			path.append(new_pt)
			path_len += 1
			# this version is scaled
			if classifier.predict_single(new_pt, env.scaler, env.classifier) == 1:
				transit += 1
				print(original_individual, f"successful: {new_pt}",  path_len, path)
				return transit, path_len, env.distance_to_closest_k_points(new_pt)		# the last term gives the Knn distance from k nearest points
			else:
				number = env.states[new_pt]
				if (new_pt == individual):
					print("unsuccessful1: ", original_individual)
					return transit, path_len, env.distance_to_closest_k_points(new_pt)
				individual = new_pt
		else:
			print("unsuccessful2: ", original_individual)
			return transit, path_len, env.distance_to_closest_k_points(new_pt)

	undesirable_x = []
	for no, i in enumerate(env.dataset.to_numpy()):
	# for no, i in enumerate(X_test.to_numpy()):
		if classifier.predict_single(i, env.scaler, env.classifier) == 0:
			undesirable_x.append(tuple(i))
	
	print(len(undesirable_x), "Total points to run the approach on")
	if len(undesirable_x) == 0:
		return 
	
	successful_transitions = 0
	total_path_len = 0
	knn_dist = 0
	# import ipdb; ipdb.set_trace()
	st = time.time()
	for no_, individual in enumerate(undesirable_x):
		transit, path_length, single_knn_dist = return_counterfactual(individual, successful_transitions)
		if transit > successful_transitions:
			successful_transitions = transit
			total_path_len += path_length
			knn_dist += single_knn_dist

	try:
		avg_path_len = total_path_len / successful_transitions
		avg_knn_dist = knn_dist / successful_transitions
		print(successful_transitions, len(undesirable_x), avg_path_len, avg_knn_dist)
	except:		# due to zero division error 
		pass

	success_rate = successful_transitions / len(undesirable_x)
	return success_rate, avg_path_len, avg_knn_dist, time.time() - st


if __name__ == "__main__":
	clf, dataset, scaler, X_test = classifier.train_model(1)	
	# n_actions = 2 * (len(dataset.columns) - 1)		# target is removed
	n_actions = 2 * len(dataset.columns)
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
	example = 1

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
		env = environment(n_states, n_actions, clf, gamma, scaler, dataset) #, X_train=X_train, closest_points=closest_points, dist_lambda=dist_lambda)
		final_policy, V = policy_improvement()
		np.save(f'final_policy_example{example}.npy', final_policy) # save
		np.save(f'value_example{example}.npy', V) # save
		# final_policy = np.load(f'final_policy_example{example}.npy')
		# V = np.load(f'value_example{example}.npy')
		percentage_success, avg_path_len, avg_knn_dist, time_taken = use_policy(final_policy, V, X_test)
	
	with open("expt_results.csv", "a") as f:
		print(f"Example{example}, {percentage_success}, {avg_path_len}, {avg_knn_dist}, {time_taken}, None", file=f)
	print("DONE")
