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


class environment:
    def __init__(self, n_actions, clf, X_train=None, closest_points=None, dist_lambda=0):
        self.observation_space = np.zeros([3])  # just created for shape[0] = 3
        self.action_space = n_actions
        # self.action_space.n = n_actions
        # self.nS = n_states
        self.nA = n_actions
        self.classifier = clf
        # self.states = {}
        self.state_count = 0
        self.X_train = X_train
        self.previous_state = np.zeros([3])
        self.current_state = np.zeros([3])  # we will update this as the agent explores
        self.no_points = closest_points
        self.dist_lambda = dist_lambda
        # self.P1 = np.zeros((self.nS, self.nA))
        # self.P = [[0 for i in range(self.nA)] for j in range(self.nS)]
        # import ipdb; ipdb.set_trace()
        # self.action_map = {'a1': 0, 'a2': 1, 'b1': 2, 'b2': 3, 'c1': 4} #, 'c2': 5}
        # self.reverse_action_map = {0: 'a1', 1: 'a2', 2: 'b1', 3: 'b2', 4: 'c1'} #, 5: 'c2'}
        self.reverse_action_map = {0: 'a1', 1: 'a2', 2: 'b1', 3: 'c1'} #, 4: 'c1'} #, 5: 'c2'}
        self.action_map = {'a1': 0, 'a2': 1, 'b1': 2, 'c1': 3} #, 'c1': 4} #, 'c2': 5}
        # state_sequence = 1 * a + 10 * b + 100 * c

        # Let's do Knn only on the second and third feature because first is random
        self.knn = NearestNeighbors(n_neighbors=6, p=2)	# 1 would be self
        self.knn.fit(self.X_train.drop(columns = ['a']))
        # import ipdb; ipdb.set_trace()

    def step(self, action):
        a, b, c = self.current_state
        self.previous_state = copy.deepcopy(self.current_state)
        if self.reverse_action_map[action] == 'a1':
            if a <= 3:
                # self.P[present_state][self.action_map['a1']] = [(1.0, self.state_sequence(a+1, b, c), self.model(a+1, b, c) - self.model(a, b, c) - 1 - self.dist_lambda * self.distance_to_closest_k_points(a+1, b, c), None)]
                self.current_state = np.array([a+1, b, c])
                done = self.prediction(self.current_state)
                # return self.current_state, self.model(self.current_state) - self.model(a, b, c) - 1 - self.dist_lambda * self.distance_to_closest_k_points(self.current_state), done
            else:
                # self.P[present_state][self.action_map['a1']] = [(1.0, present_state, -10, None)]
                done = self.prediction(self.current_state)
                return self.current_state, -10, done, None
        if self.reverse_action_map[action] == 'a2':
            if a >= 1:
                # self.P[present_state][self.action_map['a2']] = [(1.0, self.state_sequence(a-1, b, c), self.model(a-1, b, c) - self.model(a, b, c) - 1 - self.dist_lambda * self.distance_to_closest_k_points(a-1, b, c), None)]
                self.current_state = np.array([a-1, b, c])
                done = self.prediction(self.current_state)
            else:
                # self.P[present_state][self.action_map['a2']] = [(1.0, present_state, -10, None)]
                done = self.prediction(self.current_state)
                return self.current_state, -10, done, None
        if self.reverse_action_map[action] == 'b1':
            if b <= 3:
                # self.P[present_state][self.action_map['b1']] = [(1.0, self.state_sequence(a, b+1, c), self.model(a, b+1, c) - self.model(a, b, c) - 1 - self.dist_lambda * self.distance_to_closest_k_points(a, b+1, c), None)]
                self.current_state = np.array([a, b+1, c])
                done = self.prediction(self.current_state)
            else:
                # self.P[present_state][self.action_map['b1']] = [(1.0, present_state, -10, None)]		# 10 times the negative reward if the agent tries to violate the boundary condition
                done = self.prediction(self.current_state)
                return self.current_state, -10, done, None
        # if b >= 1:
        # 	self.P[present_state][self.action_map['b2']] = [(1.0, self.state_sequence(a, b-1, c), self.model(a, b-1, c) - self.model(a, b, c) - 1 - self.dist_lambda * self.distance_to_closest_k_points(a, b-1, c), None)]
        # else:
        # 	self.P[present_state][self.action_map['b2']] = [(1.0, present_state, -10, None)]
        if self.reverse_action_map[action] == 'c1':
            if c <= 3:
                # self.P[present_state][self.action_map['c1']] = [(1.0, self.state_sequence(a, b, c+1), self.model(a, b, c+1) - self.model(a, b, c) - 1 - self.dist_lambda * self.distance_to_closest_k_points(a, b, c+1), None)]
                self.current_state = np.array([a, b, c+1])
                done = self.prediction(self.current_state)
            else:
                # self.P[present_state][self.action_map['c1']] = [(1.0, present_state, -10, None)]
                done = self.prediction(self.current_state)
                return self.current_state, -10, done, None
        # no more action c2
        # if c >= 1:
        # 	self.P[present_state][self.action_map['c2']] = [(1.0, self.state_sequence(a, b, c-1), self.model(a, b, c-1) - self.model(a, b, c) - 1 - self.dist_lambda * self.distance_to_closest_k_points(a, b, c-1), None)]
        # else:
        # 	self.P[present_state][self.action_map['c2']] = [(1.0, present_state, -10, None)]
        # print(a, b, c, "hello")
        # print("done")
        return self.current_state, self.model(self.current_state) - self.model(self.previous_state) - 1 - self.dist_lambda * self.distance_to_closest_k_points(self.current_state), done, None

    def state_sequence(self, a, b, c):
        if not (a, b, c) in self.states:
            self.states[(a, b, c)] = self.state_count
            self.state_count += 1

        return self.states[(a, b, c)]

    def model(self, pt):
        return self.classifier.predict_proba(pt.reshape(1,-1))[0][1]		# find the probability of belonging to class 1 - 
        # There was a major bug in this line, instead of x,y,z, I had written x,y,x. 

    def distance_to_closest_k_points(self, pt):
        x, y, z = pt
        nearest_dist, nearest_points = self.knn.kneighbors(np.array([y,z]).reshape(1,-1), self.no_points, return_distance=True)		# we will take the 5 closest points. We don't need 6 here because the input points are not training pts.
        # quantity = np.mean(nearest_dist) / self.no_points
        quantity = np.mean(nearest_dist)		# Now we have that lambda, so no need to divide by self.no_points 
        # print((x,y,z), quantity)
        return quantity

    def reset(self):
        self.current_state = self.X_train.sample().to_numpy()[0]
        return self.current_state

    def prediction(self, pt):
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


def update_policy(policy):
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


def main(episodes, env, policy):
    running_reward = 10
    for episode in range(episodes):
        # import ipdb; ipdb.set_trace()
        state = env.reset()  # Reset environment and record the starting state
        done = False

        for time in range(100):
            action = select_action(policy, state)
            # Step through environment using chosen action
            # state, reward, done, _ = env.step(action.data[0])
            state, reward, done, _ = env.step(action.item())

            # Save reward
            policy.reward_episode.append(reward)
            if done:
                break

        # Used to determine when the environment is solved.
        running_reward = (running_reward * 0.99) + (time * 0.01)

        policy = update_policy(policy)

        if episode % 50 == 0:
            print(f'Episode {episode}\tLast length: {time:5d}\tAverage length: {running_reward:.2f}')

        # I can remove this, because I don't have a sense of upper threshold on the reward
        # if running_reward > env.spec.reward_threshold:
        #     print(f"Solved! Running reward is now {running_reward} and the last episode runs to {time} time steps!")
        #     break
    return policy


def plot(episodes):
    import matplotlib.pyplot as plt
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


def use_policy(policy, env, file):
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
        # number = env.state_sequence(*individual)

        maxtry = 30
        attempt_no = 0
        while attempt_no < maxtry:
            individual = torch.from_numpy(individual).type(torch.FloatTensor)
            actions = policy(Variable(individual))
            distribution = Categorical(actions)
            action = distribution.sample()
            # action_ = np.where(policy[number] == 1)[0]
            # assert len(action_) == 1
            action = env.reverse_action_map[action.item()]
            new_pt = np.array(take_action(*individual, action))
            cost += 1
            attempt_no += 1
            if new_pt.any() == None:       # happens for illegal actions, for eg. a1 at [4,4,4]
                print("unsuccessful: ", original_individual)
                return transit, cost, None
            if env.prediction(new_pt.reshape(1, -1)):       # if this is equal to 1
                transit += 1
                print(original_individual, f"successful: {new_pt}",  cost)
                # total_cost += cost
                return transit, cost, env.distance_to_closest_k_points(new_pt)		# the last term gives the Knn distance from k nearest points
            else:
                # number = env.state_sequence(*new_pt)
                if (new_pt == individual):
                    print("unsuccessful: ", original_individual)
                    return transit, cost, None
                individual = new_pt
        else:
            print("unsuccessful: ", original_individual)
            return transit, cost, None

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


def create_synthetic_data(file):
    # import ipdb; ipdb.set_trace()
    x1 = np.linspace(-3*np.pi, 3*np.pi, 201)
    x2 = np.sin(x1)
    # return x1, x2
    y = x1 > 0
    graph_nodes_count = 3
    graph_data = np.zeros( (x1.shape[0], graph_nodes_count)  )
    graph_data[:, 0] = x1
    graph_data[:, 1] = x2
    graph_data[:, 2] = y
    graph_data = pd.DataFrame(graph_data, columns=['x1', 'x2', 'y'] )
    graph_data.to_csv(file, index=False)
    # plt.plot(x1, x2)
    # plt.xlabel('Angle [rad]')
    # plt.ylabel('sin(x)')
    # plt.axis('tight')
    # plt.savefig('sin_curve.png')
    


# Hyperparameters
learning_rate = 0.1
gamma = 0.99
# file = "synthetic_snake.csv"
file = "synthetic2.csv"
if not os.path.exists(file):
    create_synthetic_data(file)
    exit(0)

clf, X_train = train_model(file)
# env = gym.make('CartPole-v1')
n_actions = 4
env = environment(n_actions=n_actions, clf=clf, X_train=X_train)
# env.seed(1)

policy = Policy(env, gamma)

optimizer = optim.Adam(policy.parameters(), lr=learning_rate)
episodes = 1001
final_policy = main(episodes, env, policy)
# print(final_policy, "Done")
# import ipdb; ipdb.set_trace()
percentage_success, avg_cost, knn_dist = use_policy(final_policy, env, file)
