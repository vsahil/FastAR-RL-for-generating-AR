import numpy as np 
import pandas as pd 
import sys, os, copy
from sklearn.neural_network import MLPClassifier
# from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
import random
from sklearn.neighbors import NearestNeighbors


"""
__Args__:
    1. policy: [S, A] shaped matrix representing the policy.
    2. env: OpenAI env.
        i.   env.P transition probabilities of the environment.
        ii.  env.P[s][a] is a list of transition tuples P[s][a] == [(probability, nextstate, reward, done), ...].
        iii. env.nS is a number of states in the environment.
        iv.  env.nA is a number of actions in the environment.
    3. discount_factor: Gamma discount factor.
    4. theta: We stop evaluation once our value function change is less than theta for all states.

__Returns__:
    Vector of length env.nS representing the value function.
    Matrix of length env.nS x env.nA representing the policy.
    """
    
# MDP env:
#     - env.nS = 16 
#       s ∈ S = {0...15}, where 0 and 15 are terminal states
#     - env.nA = 4, 
#       a ∈ A = {UP = 0,RIGHT = 1,DOWN = 2,LEFT = 3} 
#     - P[s][a]= {P[s][UP],P[s][RIGHT], P[s][DOWN], P[s][LEFT]}  : state transition function specifying P(ns_up|s,UP). 
#       i.e. P[s][UP] = [(1.0, ns_up, reward, is_done(ns_up))] # Not a terminal state
#            P[s][UP] = [(1.0, s, reward, True)] # A terminal state
#     - R is a reward function R(s,a,s')
#       reward = 0.0 if we are stuck in a terminal state, else -1.0


# Local Variables
# - policy[s]: action array
# - chosen_a: a real number ∈ {0,1,2,3}
# - action_values: action values array
# - best_a: integer variable 
# - policy_stable: boolean variable 
# - delta: integer variable
# - V[S]: real array 


class environment:
	def __init__(self, n_states, n_actions, clf, X_train=None, closest_points=None):
		self.nS = n_states
		self.nA = n_actions
		self.classifier = clf
		self.states = {}
		self.state_count = 0
		self.X_train = X_train
		self.no_points = closest_points
		self.P1 = np.zeros((self.nS, self.nA))
		self.P = [[0 for i in range(self.nA)] for j in range(self.nS)]
		# import ipdb; ipdb.set_trace()
		self.action_map = {'a1': 0, 'a2': 1, 'b1': 2, 'b2': 3, 'c1': 4} #, 'c2': 5}
		self.reverse_action_map = {0: 'a1', 1: 'a2', 2: 'b1', 3: 'b2', 4: 'c1'} #, 5: 'c2'}
		# state_sequence = 1 * a + 10 * b + 100 * c

		# Let's do Knn only on the second and third feature because first is random
		self.knn = NearestNeighbors(n_neighbors=6, p=2)	# 1 would be self
		self.knn.fit(self.X_train.drop(columns = ['a']))
		# import ipdb; ipdb.set_trace()

		for a in range(5):
			for b in range(5):
				for c in range(5):
					present_state = self.state_sequence(a, b, c)
					if a <= 3:
						self.P[present_state][self.action_map['a1']] = [(1.0, self.state_sequence(a+1, b, c), self.model(a+1, b, c) - self.model(a, b, c) - 1 - self.distance_to_closest_k_points(a+1, b, c), None)]
					else:
						self.P[present_state][self.action_map['a1']] = [(1.0, present_state, -10, None)]
					if a >= 1:
						self.P[present_state][self.action_map['a2']] = [(1.0, self.state_sequence(a-1, b, c), self.model(a-1, b, c) - self.model(a, b, c) - 1 - self.distance_to_closest_k_points(a-1, b, c), None)]
					else:
						self.P[present_state][self.action_map['a2']] = [(1.0, present_state, -10, None)]
					if b <= 3:
						self.P[present_state][self.action_map['b1']] = [(1.0, self.state_sequence(a, b+1, c), self.model(a, b+1, c) - self.model(a, b, c) - 1 - self.distance_to_closest_k_points(a, b+1, c), None)]
					else:
						self.P[present_state][self.action_map['b1']] = [(1.0, present_state, -10, None)]		# 10 times the negative reward if the agent tries to violate the boundary condition
					if b >= 1:
						self.P[present_state][self.action_map['b2']] = [(1.0, self.state_sequence(a, b-1, c), self.model(a, b-1, c) - self.model(a, b, c) - 1 - self.distance_to_closest_k_points(a, b-1, c), None)]
					else:
						self.P[present_state][self.action_map['b2']] = [(1.0, present_state, -10, None)]
					if c <= 3:
						self.P[present_state][self.action_map['c1']] = [(1.0, self.state_sequence(a, b, c+1), self.model(a, b, c+1) - self.model(a, b, c) - 1 - self.distance_to_closest_k_points(a, b, c+1), None)]
					else:
						self.P[present_state][self.action_map['c1']] = [(1.0, present_state, -10, None)]
					# no more action c2
					# if c >= 1:
					# 	self.P[present_state][self.action_map['c2']] = [(1.0, self.state_sequence(a, b, c-1), self.model(a, b, c-1) - self.model(a, b, c) - 1 - self.distance_to_closest_k_points(a, b, c-1), None)]
					# else:
					# 	self.P[present_state][self.action_map['c2']] = [(1.0, present_state, -10, None)]
					# print(a, b, c, "hello")
		# print("done")


	def state_sequence(self, a, b, c):
		if not (a, b, c) in self.states:
			self.states[(a, b, c)] = self.state_count
			self.state_count += 1

		return self.states[(a, b, c)]
	
	
	def model(self, x, y, z):
		return self.classifier.predict_proba(np.array([x,y,z]).reshape(1,-1))[0][1]		# find the probability of belonging to class 1 - 
		# There was a major bug in this line, instead of x,y,z, I had written x,y,x. 


	def distance_to_closest_k_points(self, x, y, z):
		nearest_dist, nearest_points = self.knn.kneighbors(np.array([y,z]).reshape(1,-1), self.no_points, return_distance=True)		# we will take the 5 closest points. We don't need 6 here because the input points are not training pts.
		quantity = np.mean(nearest_dist) / self.no_points
		# print((x,y,z), quantity)
		return quantity



def policy_eval(policy, env, discount_factor=1.0, theta=0.00001, max_iterations=1000):
    # Initialize the value function
	V = np.zeros(env.nS)
    # While our value function is worse than the threshold theta
	# import ipdb; ipdb.set_trace()
	iters = 0
	# while iters < max_iterations:
	while True:
        # Keep track of the update done in value function
		delta = 0
        # For each state, look ahead one step at each possible action and next state
		for s in range(env.nS):
			v = 0
			# The possible next actions, policy[s]:[a, action_prob]
			for a, action_prob in enumerate(policy[s]): 
				# For each action, look at the possible next states, 
				for prob, next_state, reward, done in env.P[s][a]: # state transition P[s][a] == [(prob, nextstate, reward, done), ...]
					# Calculate the expected value function
					v += action_prob * prob * (reward + discount_factor * V[next_state]) # P[s, a, s']*(R(s,a,s')+γV[s'])
					# How much our value function changed across any states .  
			delta = max(delta, np.abs(v - V[s]))
			V[s] = v
		# Stop evaluating once our value function update is below a threshold
		# print(iters, "done", delta)
		if delta < theta:
			break
		iters += 1
		
	return np.array(V)


def policy_improvement(env, policy_eval_fn=policy_eval, discount_factor=1.0):
    # Initiallize a policy arbitarily
    policy = np.ones([env.nS, env.nA]) / env.nA			# initial same probability of taking all actions

    while True:
        # Compute the Value Function for the current policy
        V = policy_eval_fn(policy, env, discount_factor)

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
            best_a = np.argmax(action_values)

            # Greedily (max in the above line) update the policy
            if chosen_a != best_a:
                policy_stable = False
            policy[s] = np.eye(env.nA)[best_a]		# so this given the one-hot vector of policy for each state

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


def train_model():
	# import ipdb; ipdb.set_trace()
	# X, y = make_classification(n_samples=100, random_state=1)
	total_dataset = pd.read_csv("synthetic.csv")
	Y = total_dataset['y']
	total_dataset = total_dataset.drop(columns=['y'])
	X_train, X_test, y_train, y_test = train_test_split(total_dataset, Y, stratify=Y, random_state=1)
	clf = MLPClassifier(random_state=1, max_iter=300).fit(X_train, y_train)
	# clf.predict_proba(X_test[:1])
	return clf, X_train
	# clf.predict(X_test[:5, :])

	clf.score(X_test, y_test)


def use_policy(policy, classifier, env):
	def take_action(a, b, c, action):
		if action == "a1" and a <= 3:
			return (a+1, b, c)
		elif action == "a2" and a >= 1:
			return (a-1, b, c)
		elif action == "b1" and b <= 3:
			return (a, b+1, c)
		elif action == "b2" and b >= 1:
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
				return transit, cost
			else:
				number = env.state_sequence(*new_pt)
				if (new_pt == individual).all():
					print("unsuccessful: ", original_individual)
					return transit, cost
				individual = new_pt
		else:
			print("unsuccessful: ", original_individual)
			return transit, cost


	total_dataset = pd.read_csv("synthetic.csv")
	Y = total_dataset['y']
	total_dataset = total_dataset.drop(columns=['y'])
	X_train, X_test, y_train, y_test = train_test_split(total_dataset, Y, stratify=Y, random_state=1)
	undesirable_x = X_test[y_test == 0].to_numpy()
	successful_transitions = 0
	total_cost = 0
	for no_, individual in enumerate(undesirable_x):
		transit, cost = return_counterfactual(individual, successful_transitions)
		if transit > successful_transitions:
			successful_transitions = transit
			total_cost += cost

	avg_cost = total_cost / successful_transitions
	print(successful_transitions, len(undesirable_x), avg_cost)			
	# print("see")
	return successful_transitions * 100.0 / len(undesirable_x)


from utils import create_pdf_from_table

if __name__ == "__main__":
	file = "synthetic2.csv"
	if not os.path.exists(file):
		create_synthetic_data(file)
		exit(0)
	clf, X_train = train_model()
	n_actions = 5 		# choose a state, increment a state value by 1 or decrement by 1, 3*2 = 6 actions, except that now age can't decrease. 
	n_states = 125 		# 5 values, each for the 3 states, 5*5*5 = 125. 
	closest_points = 5
	gamma = 0.5
	full_result_table = []
	seq_gamma = [0.1, 0.2, 0.5, 0.8, 0.9]
	seq_closest_points = [1, 2, 5, 10]

	for gamma in seq_gamma:
		result_for_gamma = []
		for closest_points in seq_closest_points:
			env = environment(n_states, n_actions, clf, X_train=X_train, closest_points=closest_points)
			# import ipdb; ipdb.set_trace()
			final_policy, V = policy_improvement(env, discount_factor=gamma)
			percentage_success = use_policy(final_policy, clf, env)
			result_for_gamma.append(percentage_success)
		print(gamma, result_for_gamma, "done")
		full_result_table.append(result_for_gamma)
	
	# import ipdb; ipdb.set_trace()
	headers = ["Closest Points", "Gamma = .1", "Gamma = .2", "Gamma = .5", "Gamma = .8", "Gamma = .9"]
	table = np.array([seq_closest_points, np.around(np.array(full_result_table[0]), decimals=3), np.around(np.array(full_result_table[1]), decimals=3), np.around(np.array(full_result_table[2]), decimals=3), np.around(np.array(full_result_table[3]), decimals=3), np.around(np.array(full_result_table[4]), decimals=3)]).T
	create_pdf_from_table('success_causal_dist_k', table, headers)
	print("DONE")