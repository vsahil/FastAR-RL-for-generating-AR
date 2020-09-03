# import gym
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm, trange
import pandas as pd
import sys, os, copy, math
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
import random
from sklearn.neighbors import NearestNeighbors
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
from torch.distributions import Categorical
# %matplotlib inline
torch.manual_seed(1)

# This has non-deterministic actions - because we sample from probability. 
# not working as expected - I might wanna inspect, change L2 distance to L1. Intead of sampling points from sin curve, sample any point from the left half of the plot area, will generalize better. 
# Find the 2, 5, 10 closest points and plot it for some random points and see if it captures what you want to capture. 


class environment:
    def __init__(self, n_actions, clf, X_train=None, closest_points=None, dist_lambda=0, file=None):
        self.observation_space = np.zeros([X_train.shape[1]])  # just created for shape[0] = 3
        self.action_space = n_actions
        # self.action_space.n = n_actions
        # self.nS = n_states
        self.nA = n_actions
        self.classifier = clf
        # self.states = {}
        self.state_count = 0
        self.X_train = X_train
        self.previous_state = np.zeros([X_train.shape[1]])
        self.current_state = np.zeros([X_train.shape[1]])  # we will update this as the agent explores
        self.no_points = closest_points         # when this is None, it defaults to 5.
        self.dist_lambda = dist_lambda
        if file:
            self.total_dataset = pd.read_csv(file)
        # self.P1 = np.zeros((self.nS, self.nA))
        # self.P = [[0 for i in range(self.nA)] for j in range(self.nS)]
        # import ipdb; ipdb.set_trace()
        # self.action_map = {'a1': 0, 'a2': 1, 'b1': 2, 'b2': 3, 'c1': 4} #, 'c2': 5}
        # self.reverse_action_map = {0: 'a1', 1: 'a2', 2: 'b1', 3: 'b2', 4: 'c1'} #, 5: 'c2'}
        # self.reverse_action_map = {0: 'a1', 1: 'a2', 2: 'b1', 3: 'c1'} #, 4: 'c1'} #, 5: 'c2'}
        # self.action_map = {'a1': 0, 'a2': 1, 'b1': 2, 'c1': 3} #, 'c1': 4} #, 'c2': 5}

        self.reverse_action_map = {0: 'n', 1: 's', 2: 'e', 3: 'w'}
        self.action_map = {'n': 0, 's': 1, 'e': 2, 'w': 3}
        
        self.min_x1 = -0.5
        self.max_x1 = 5
        # x1 = np.linspace(min_x1, max_x1, 201)
        # x2 = np.sin(x1 / 3)
        self.min_x2 = np.sin(self.min_x1 / 3)
        self.max_x2 = np.sin(self.max_x1 / 3)
        
        # Let's do Knn only on the second and third feature because first is random - no dropping in the snake dataset
        self.knn = NearestNeighbors(n_neighbors=25, p=2)	# 1 would be self
        # self.knn.fit(self.X_train.drop(columns = ['a']))
        self.knn.fit(self.X_train)
        # import ipdb; ipdb.set_trace()

    # def step(self, action):
    #     a, b, c = self.current_state
    #     self.previous_state = copy.deepcopy(self.current_state)
    #     if self.reverse_action_map[action] == 'a1':
    #         if a <= 3:
    #             # self.P[present_state][self.action_map['a1']] = [(1.0, self.state_sequence(a+1, b, c), self.model(a+1, b, c) - self.model(a, b, c) - 1 - self.dist_lambda * self.distance_to_closest_k_points(a+1, b, c), None)]
    #             self.current_state = np.array([a+1, b, c])
    #             done = self.prediction(self.current_state)
    #             # return self.current_state, self.model(self.current_state) - self.model(a, b, c) - 1 - self.dist_lambda * self.distance_to_closest_k_points(self.current_state), done
    #         else:
    #             # self.P[present_state][self.action_map['a1']] = [(1.0, present_state, -10, None)]
    #             done = self.prediction(self.current_state)
    #             return self.current_state, -10, done, None
    #     if self.reverse_action_map[action] == 'a2':
    #         if a >= 1:
    #             # self.P[present_state][self.action_map['a2']] = [(1.0, self.state_sequence(a-1, b, c), self.model(a-1, b, c) - self.model(a, b, c) - 1 - self.dist_lambda * self.distance_to_closest_k_points(a-1, b, c), None)]
    #             self.current_state = np.array([a-1, b, c])
    #             done = self.prediction(self.current_state)
    #         else:
    #             # self.P[present_state][self.action_map['a2']] = [(1.0, present_state, -10, None)]
    #             done = self.prediction(self.current_state)
    #             return self.current_state, -10, done, None
    #     if self.reverse_action_map[action] == 'b1':
    #         if b <= 3:
    #             # self.P[present_state][self.action_map['b1']] = [(1.0, self.state_sequence(a, b+1, c), self.model(a, b+1, c) - self.model(a, b, c) - 1 - self.dist_lambda * self.distance_to_closest_k_points(a, b+1, c), None)]
    #             self.current_state = np.array([a, b+1, c])
    #             done = self.prediction(self.current_state)
    #         else:
    #             # self.P[present_state][self.action_map['b1']] = [(1.0, present_state, -10, None)]		# 10 times the negative reward if the agent tries to violate the boundary condition
    #             done = self.prediction(self.current_state)
    #             return self.current_state, -10, done, None
    #     # if b >= 1:
    #     # 	self.P[present_state][self.action_map['b2']] = [(1.0, self.state_sequence(a, b-1, c), self.model(a, b-1, c) - self.model(a, b, c) - 1 - self.dist_lambda * self.distance_to_closest_k_points(a, b-1, c), None)]
    #     # else:
    #     # 	self.P[present_state][self.action_map['b2']] = [(1.0, present_state, -10, None)]
    #     if self.reverse_action_map[action] == 'c1':
    #         if c <= 3:
    #             # self.P[present_state][self.action_map['c1']] = [(1.0, self.state_sequence(a, b, c+1), self.model(a, b, c+1) - self.model(a, b, c) - 1 - self.dist_lambda * self.distance_to_closest_k_points(a, b, c+1), None)]
    #             self.current_state = np.array([a, b, c+1])
    #             done = self.prediction(self.current_state)
    #         else:
    #             # self.P[present_state][self.action_map['c1']] = [(1.0, present_state, -10, None)]
    #             done = self.prediction(self.current_state)
    #             return self.current_state, -10, done, None
    #     # no more action c2
    #     # if c >= 1:
    #     # 	self.P[present_state][self.action_map['c2']] = [(1.0, self.state_sequence(a, b, c-1), self.model(a, b, c-1) - self.model(a, b, c) - 1 - self.dist_lambda * self.distance_to_closest_k_points(a, b, c-1), None)]
    #     # else:
    #     # 	self.P[present_state][self.action_map['c2']] = [(1.0, present_state, -10, None)]
    #     # print(a, b, c, "hello")
    #     # print("done")
    #     return self.current_state, self.model(self.current_state) - self.model(self.previous_state) - 1 - self.dist_lambda * self.distance_to_closest_k_points(self.current_state), done, None

    def step(self, action):
        self.north_magnitude = 0.05
        self.south_magnitude = -0.05
        self.east_magnitude = 0.1
        self.west_magnitude = -0.1
        
        x1, x2 = self.current_state
        self.previous_state = copy.deepcopy(self.current_state)
        if self.reverse_action_map[action] == 'n':
            if x2 < 1:
                self.current_state = np.array([x1, x2 + self.north_magnitude])
                done = self.prediction(self.current_state)
            else:
                done = self.prediction(self.current_state)
                return self.current_state, -10, done, None

        if self.reverse_action_map[action] == 's':
            if x2 > -1:
                self.current_state = np.array([x1, x2 + self.south_magnitude])
                done = self.prediction(self.current_state)
            else:
                done = self.prediction(self.current_state)
                return self.current_state, -10, done, None

        if self.reverse_action_map[action] == 'e':
            self.current_state = np.array([x1 + self.east_magnitude, x2])
            done = self.prediction(self.current_state)

        if self.reverse_action_map[action] == 'w':
            self.current_state = np.array([x1 + self.west_magnitude, x2])
            done = self.prediction(self.current_state)
        # import ipdb; ipdb.set_trace()
        return self.current_state, self.model(self.current_state) - self.model(self.previous_state) - 1 - self.dist_lambda * self.distance_to_closest_k_points(self.current_state), done, None

    def model(self, pt):
        # Let the probability in the negative x zone be the y point of the straight line connecting (-8, 0) and (0, 1), which is stating that at x= -8, reward is 0 and at x = 0, reward is 1
        x1, x2 = pt
        if x1 >= 5:      # simple classifier
            return 1      # end of episode high reward
        # return x1/8 + 1     # this will change 
        # return 0
        return x1/5
        return self.classifier.predict_proba(pt.reshape(1, -1))[0][1]		# find the probability of belonging to class 1 - 
        # There was a major bug in this line, instead of x,y,z, I had written x,y,x. 

    def distance_to_closest_k_points(self, pt):
        # x, y, z = pt
        x1, x2 = pt
        nearest_dist, nearest_points = self.knn.kneighbors(np.array([x1, x2]).reshape(1,-1), self.no_points, return_distance=True)		# we will take the 5 closest points. We don't need 6 here because the input points are not training pts.
        # quantity = np.mean(nearest_dist) / self.no_points
        quantity = np.mean(nearest_dist)		# Now we have that lambda, so no need to divide by self.no_points 
        # print((x,y,z), quantity)
        return quantity

    def reset(self):
        # I think we should only sample X_train whose y == 0, this increased learning by a lot. The other way could have been to increase the number of episodes, but this is more effective
        # 
        # self.current_state = self.total_dataset[self.total_dataset['y'] == 0.0].sample().to_numpy()[0][:2]   # drop y 
        # now instead, sample any point from the left half of the 2-D plot.
        # import ipdb; ipdb.set_trace()
        x1_sample = (self.max_x1 - self.min_x1) * np.random.random_sample() + self.min_x1
        x2_sample = (self.max_x2 - self.min_x2) * np.random.random_sample() + self.min_x2
        self.current_state = np.array([x1_sample, x2_sample])
        # self.current_state = self.X_train.sample().to_numpy()[0]
        return self.current_state

    def prediction(self, pt):
        # import ipdb; ipdb.set_trace()
        # print(pt, "hello")
        try:
            x1, x2 = pt
        except:
            # if pt.shape == (1, 2):
            x1, x2 = pt[0]
        if x1 >= 5:      # simple classifier
            return True
        return False
        return self.classifier.predict(pt.reshape(1, -1))[0] == 1


class Policy(nn.Module):
    def __init__(self, env, gamma):
        super(Policy, self).__init__()
        self.state_space = env.observation_space.shape[0]
        # self.action_space = env.action_space.n
        self.action_space = env.action_space

        self.l1 = nn.Linear(self.state_space, 128, bias=False)
        self.l2 = nn.Linear(128, self.action_space, bias=False)

        self.gamma = gamma

        # Episode policy and reward history
        # self.policy_history = Variable(torch.Tensor())
        self.policy_history = torch.Tensor([])
        self.reward_episode = []
        # Overall reward and loss history
        self.reward_history = []
        self.loss_history = []

    def forward(self, x):
        model = torch.nn.Sequential(
            self.l1,
            nn.Dropout(p=0.6),
            nn.ReLU(),
            self.l2,
            nn.Softmax(dim=-1)
        )
        return model(x)


def select_action(policy, state):
    # Select an action (0 or 1) by running policy model and choosing based on the probabilities in state
    state = torch.from_numpy(state).type(torch.FloatTensor)
    actions = policy(Variable(state))
    distribution = Categorical(actions)
    action = distribution.sample()

    # Add log probability of our chosen action to our history
    # if policy.policy_history.dim() != 0:
    #     policy.policy_history = torch.cat([policy.policy_history, c.log_prob(action)])
    # else:
    #     policy.policy_history = (c.log_prob(action))

    # Add log probability of our chosen action to our history
    policy.policy_history = torch.cat([
        policy.policy_history,
        distribution.log_prob(action).reshape(1)
    ])

    return action


def update_policy(policy, optimizer):
    R = 0
    rewards = []
    # import ipdb; ipdb.set_trace()
    # Discount future rewards back to the present using gamma
    for r in policy.reward_episode[::-1]:
        R = r + policy.gamma * R
        rewards.insert(0, R)

    # Scale rewards
    rewards = torch.FloatTensor(rewards)
    if not math.isnan(rewards.std()):
        rewards = (rewards - rewards.mean()) / (rewards.std() + np.finfo(np.float32).eps)
    else:
        assert rewards.mean() == rewards   # one case I know if when rewards is an one item vector
        rewards = (rewards - rewards.mean()) / (np.finfo(np.float32).eps)
        assert rewards.item() == 0

    # Calculate loss
    loss = (torch.sum(torch.mul(policy.policy_history, Variable(rewards)).mul(-1), -1))

    # Update network weights
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # Save and intialize episode history counters
    # policy.loss_history.append(loss.data[0])
    policy.loss_history.append(loss.item())
    policy.reward_history.append(np.sum(policy.reward_episode))
    policy.policy_history = Variable(torch.Tensor())
    policy.reward_episode = []
    return policy


def main(episodes, env, policy, optimizer):
    running_reward = 10
    max_time = 1000
    for episode in range(episodes):
        # import ipdb; ipdb.set_trace()
        state = env.reset()  # Reset environment and record the starting state
        done = False
        session = [state]
        for time in range(max_time):        # 100 is enough for synthetic2.csv, 1000 for snake dataset.
            action = select_action(policy, state)
            # Step through environment using chosen action
            # state, reward, done, _ = env.step(action.data[0])
            state, reward, done, _ = env.step(action.item())

            # Save reward
            policy.reward_episode.append(reward)
            session.append((action, state))
            if done:
                break

        # Used to determine when the environment is solved.
        running_reward = (running_reward * 0.99) + (time * 0.01)

        policy = update_policy(policy, optimizer)

        if episode % 50 == 0:
            print(f'Episode {episode}\tLast length: {time:5d}\tAverage length: {running_reward:.2f}')
        # if time == max_time - 1:
        #     import ipdb; ipdb.set_trace()
        # I can remove this, because I don't have a sense of upper threshold on the reward
        # if running_reward > env.spec.reward_threshold:
        #     print(f"Solved! Running reward is now {running_reward} and the last episode runs to {time} time steps!")
        #     break
    return policy


def plot(episodes):
    window = int(episodes / 20)
    fig, ((ax1), (ax2)) = plt.subplots(2, 1, sharey=True, figsize=[9,9])
    rolling_mean = pd.Series(policy.reward_history).rolling(window).mean()
    std = pd.Series(policy.reward_history).rolling(window).std()
    ax1.plot(rolling_mean)
    ax1.fill_between(range(len(policy.reward_history)), rolling_mean - std, rolling_mean + std, color='orange', alpha=0.2)
    ax1.set_title(f'Episode Length Moving Average ({window}-episode window)')
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Episode Length')

    ax2.plot(policy.reward_history)
    ax2.set_title('Episode Length')
    ax2.set_xlabel('Episode')
    ax2.set_ylabel('Episode Length')

    fig.tight_layout(pad=2)
    plt.show()


# not a good classifier 
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


def plot_trajectories(x, success_rate, deter=False, closest_points=None, dist_lambda=None):
    # import ipdb; ipdb.set_trace()
    x1 = np.linspace(-0.5, 10, 201)
    x2 = np.sin(x1 / 3)
    # x1 = np.linspace(-3*np.pi, 3*np.pi, 201)
    # x2 = np.sin(x1)
    plt.plot(x1, x2)
    for path in x:
        xs = [i[0] for i in path]
        ys = [i[1] for i in path]
        # plt.plot(xs, ys)        # I think this will automatically change color
        markers_on = [0]        # the first point is marked with a shape
        plt.plot(xs, ys, '-D', markevery=markers_on)
    plt.xlabel('x1')
    plt.ylabel('y')
    plt.axis('tight')
    success_rate = round(success_rate, 2)
    plt.title(f'Success rate = {success_rate}, Closest pts = {closest_points}, λ = {dist_lambda}')
    if closest_points:      # is not None
        if deter:
            plt.savefig(f'plots/sine_curve/trajectories_knn_l2_deter_{closest_points}_{dist_lambda}.png')
        else:
            plt.savefig(f'plots/sine_curve/trajectories_knn_l2_nondeter_{closest_points}_{dist_lambda}.png')
    else:
        if deter:
            plt.savefig(f'plots/sine_curve/trajectories_noknn_deter.png')
        else:
            plt.savefig(f'plots/sine_curve/trajectories_noknn_nondeter.png')
    print("PLOT DONE")


def use_policy(policy, env, file, deter=False, closest_points=None, dist_lambda=None):
    # def take_action(a, b, c, action, env):
    #     if action == "a1" and a <= 3:
    #         return (a+1, b, c)
    #     elif action == "a2" and a >= 1:
    #         return (a-1, b, c)
    #     elif action == "b1" and b <= 3:
    #         return (a, b+1, c)
    #     elif action == "b2" and b >= 1:
    #         raise NotImplementedError
    #         return (a, b-1, c)
    #     elif action == "c1" and c <= 3:
    #         return (a, b, c+1)
    #     elif action == "c2" and c >= 1:
    #         raise NotImplementedError 		# c2 is not more a valid action
    #         return (a, b, c-1)

    def take_action(x1, x2, action, env):
        if action == "n" and x2 < 1:
            return (x1, x2 + env.north_magnitude)
        elif action == "s" and x2 > -1:
            return (x1, x2 + env.south_magnitude)
        elif action == "e":
            return (x1 + env.east_magnitude, x2)
        elif action == "w":
            return (x1 + env.west_magnitude, x2)

    def return_counterfactual(original_individual, transit):
        cost = 0
        individual = copy.deepcopy(original_individual)
        # number = env.state_sequence(*individual)
        # import ipdb; ipdb.set_trace()
        maxtry = 600        # we need to increase the number of tries for snake dataset, small steps is the reason, increased further for KNN loss
        attempt_no = 0
        path = [individual]
        while attempt_no < maxtry:
            individual = torch.from_numpy(individual).type(torch.FloatTensor)
            actions = policy(Variable(individual))
            distribution = Categorical(actions)
            if deter:
                action = torch.argmax(actions)      # replaced with deterministic actions
            else:
                action = distribution.sample()
            # action_ = np.where(policy[number] == 1)[0]
            # assert len(action_) == 1
            action = env.reverse_action_map[action.item()]
            new_pt = np.array(take_action(*individual, action, env))
            cost += 1
            attempt_no += 1
            # print(action, new_pt, "see")
            if new_pt.any() == None:       # happens for illegal actions, for eg. a1 at [4,4,4]
                print("unsuccessful: ", original_individual)
                return transit, cost, None, None
            else:
                path.append(new_pt)
            if env.prediction(new_pt.reshape(1, -1)):       # if this is equal to 1
                transit += 1
                print(original_individual, f"successful: {new_pt}",  cost)
                # total_cost += cost
                return transit, cost, env.distance_to_closest_k_points(new_pt), path		# the last term gives the Knn distance from k nearest points
            else:
                # number = env.state_sequence(*new_pt)
                if (new_pt == individual):
                    print("unsuccessful: ", original_individual)
                    return transit, cost, None, None
                individual = new_pt
        else:
            print("unsuccessful: ", original_individual)
            return transit, cost, None, None

    # import ipdb; ipdb.set_trace()
    total_dataset = pd.read_csv(file)
    Y = total_dataset['y']
    total_dataset = total_dataset.drop(columns=['y'])
    X_train, X_test, y_train, y_test = train_test_split(total_dataset, Y, stratify=Y, random_state=1)
    undesirable_x = X_test[y_test == 0].to_numpy()
    successful_transitions = 0
    total_cost = 0
    knn_dist = 0
    trajectories = []
    for no_, individual in enumerate(undesirable_x):
        transit, cost, single_knn_dist, path = return_counterfactual(individual, successful_transitions)
        if transit > successful_transitions:
            successful_transitions = transit
            total_cost += cost
            knn_dist += single_knn_dist
            # 0 : 0.81, 2 : 0, 5 : 0.46, 9 : 0.9, 18: -0.08
            if no_ in [0, 2, 5, 9, 18]:
                trajectories.append(path)

    try:
        avg_cost = total_cost / successful_transitions
        print(successful_transitions, len(undesirable_x), avg_cost, knn_dist)
        success_rate = successful_transitions / len(undesirable_x)
        plot_trajectories(trajectories, success_rate, deter, closest_points, dist_lambda)
        return success_rate, avg_cost, knn_dist
    except:		# due to zero division error, if all fails. 
        return None, None, None
    # print("see")


def create_synthetic_data(file):
    # import ipdb; ipdb.set_trace()
    x1 = np.linspace(-0.5, 10, 201)
    x2 = np.sin(x1 / 3)
    # return x1, x2
    y = x1 >= 5
    graph_nodes_count = 3
    graph_data = np.zeros( (x1.shape[0], graph_nodes_count)  )
    graph_data[:, 0] = x1
    graph_data[:, 1] = x2
    graph_data[:, 2] = y
    graph_data = pd.DataFrame(graph_data, columns=['x1', 'x2', 'y'] )
    graph_data.to_csv(file, index=False)
    plt.plot(x1, x2)
    plt.xlabel('Angle [rad]')
    plt.ylabel('sin(x)')
    plt.axis('tight')
    plt.savefig('sin_curve.png')


# def plot_nearest_point(env, X_train):
#     x1, x2 = pt
#     nearest_dist, nearest_points = self.knn.kneighbors(np.array([x1, x2]).reshape(1,-1), self.no_points, return_distance=True)
#     quantity = np.mean(nearest_dist)



def learn(n_actions, clf, X_train, closest_points, dist_lambda, episodes, gamma, learning_rate, file, deter):
    env = environment(n_actions=n_actions, clf=clf, X_train=X_train, 
                    closest_points=closest_points, dist_lambda=dist_lambda, file=file)

    # plot_nearest_point(env, X_train)
    policy = Policy(env, gamma)
    optimizer = optim.Adam(policy.parameters(), lr=learning_rate)
    final_policy = main(episodes, env, policy, optimizer)
    # print(final_policy, "Done")
    # import ipdb; ipdb.set_trace()
    percentage_success, avg_cost, knn_dist = use_policy(final_policy, env, file, deter, closest_points, dist_lambda)


# Hyperparameters
learning_rate = 0.01
gamma = 0.99
file = "synthetic_snake.csv"
# file = "synthetic2.csv"
if not os.path.exists(file):
    create_synthetic_data(file)
    exit(0)

clf, X_train = train_model(file)
# env = gym.make('CartPole-v1')
# import ipdb; ipdb.set_trace()
closest_points = None
dist_lambda = 0
n_actions = 4       # in the snake dataset we still have 4 actions, but they are north, south, east, west - with small magnitudes. North, south - 0.05, East, west - 0.1
episodes = 1001
deter = True

experiment = True
if experiment:
    # for closest_points in [1, 2, 5, 10]:
    #     for dist_lambda in [0.01, 0.1, 1, 10, 100, 1000]:
    closest_points = int(sys.argv[1])
    dist_lambda = float(sys.argv[2])
    print(closest_points, dist_lambda)
    learn(n_actions, clf, X_train, closest_points, dist_lambda, episodes, gamma, learning_rate, file, deter)
else:
    learn(n_actions, clf, X_train, closest_points, dist_lambda, episodes, gamma, learning_rate, file, deter)

