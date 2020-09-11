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
torch.manual_seed(1)

class environment():
    def __init__(self, n_actions, dist_lambda=0):
        self.observation_space = np.zeros([2])
        self.action_space = n_actions
        self.current_state = np.zeros([2])
        self.dist_lambda = dist_lambda
        self.reverse_action_map = {0: 'n', 1: 's', 2: 'e', 3: 'w'}
        self.action_map = {'n': 0, 's': 1, 'e': 2, 'w': 3}
        self.min_x1 = 2   # -0.5
        self.max_x1 = 7   # 5
        self.north = 0.05
        self.south = -0.05
        self.east = 0.05
        self.west = -0.05

    def step(self, action):
        x1, x2 = self.current_state
        if self.reverse_action_map[action] == 'n':
            self.current_state = np.array([x1, x2 + self.north])
            done = self.prediction(self.current_state)
        
        elif self.reverse_action_map[action] == 's':
            self.current_state = np.array([x1, x2 + self.south])
            done = self.prediction(self.current_state)
        
        elif self.reverse_action_map[action] == 'e':
            self.current_state = np.array([x1 + self.east, x2])
            done = self.prediction(self.current_state)

        elif self.reverse_action_map[action] == 'w':
            self.current_state = np.array([x1 + self.west, x2])
            done = self.prediction(self.current_state)

        reward = -(self.distance_from_manifold(self.current_state))     # - distance from manifold
        return self.current_state, reward, done
    
    def distance_from_manifold(self, point):
        return np.sqrt(np.sum(point**2))         # distance from (0, 0), that is our manifold
    
    def reset(self, set_state=True):
        x1_sample = (self.max_x1 - self.min_x1) * np.random.random_sample() + self.min_x1
        assert self.min_x1 <= x1_sample <= self.max_x1
        x2_sample = np.random.random_sample()       # samples between 0 and 1
        assert 0 <= x2_sample <= 1
        if set_state:
            self.current_state = np.array([x1_sample, x2_sample])
        return np.array([x1_sample, x2_sample])

    def prediction(self, point):
        return False            # let this be false always
    

class Policy(nn.Module):
    def __init__(self, env, gamma):
        super(Policy, self).__init__()
        self.state_space = env.observation_space.shape[0]
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
    # Discount future rewards back to the present using gamma
    for r in policy.reward_episode[::-1]:
        R = r + policy.gamma * R
        rewards.insert(0, R)

    # Scale rewards
    rewards = torch.FloatTensor(rewards)
    if not math.isnan(rewards.std()):
        rewards = (rewards - rewards.mean()) / (rewards.std() + np.finfo(np.float32).eps)
    else:
        assert rewards.mean() == rewards        # one case I know if when rewards is an one item vector
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
    max_time = 200
    for episode in range(episodes):
        state = env.reset()  # Reset environment and record the starting state
        done = False
        session = [state]
        for time in range(max_time):        # 100 is enough for synthetic2.csv, 1000 for snake dataset.
            action = select_action(policy, state)
            # Step through environment using chosen action
            state, reward, done = env.step(action.item())

            # Save reward
            policy.reward_episode.append(reward)
            if done:
                break

        # Used to determine when the environment is solved.
        running_reward = (running_reward * 0.99) + (time * 0.01)

        policy = update_policy(policy, optimizer)

        # if episode % 50 == 0:
        print(f'Episode {episode}\tLast length: {time:5d}\tAverage length: {running_reward:.2f}')
    return policy


def plot_trajectories(x, deter, dist_lambda=None):
    x1 = np.linspace(-0.5, 10, 201)
    x2 = np.sin(x1 / 3)
    plt.figure()
    plt.plot(x1, x2)
    for path in x:
        xs = [i[0] for i in path]
        ys = [i[1] for i in path]
        # plt.plot(xs, ys)        # I think this will automatically change color
        markers_on = [0]        # the first point is marked with a shape
        plt.plot(xs, ys, '-D', markevery=markers_on)
        print(len(path), "see length of trajectory")
    
    plt.xlabel('x1')
    plt.ylabel('y')
    plt.axis('tight')
    # success_rate = round(success_rate, 2)
    plt.title(f'λ = {dist_lambda}')
    if deter:
        plt.savefig(f'plots/try_follow_sine/trajectories_l2_deter_{dist_lambda}.png')
    else:
        plt.savefig(f'plots/try_follow_sine/trajectories_l2_nondeter_{dist_lambda}.png')
    print("PLOT DONE")


def use_policy(policy, env, deter):

    def take_action(x1, x2, action, env):
        if action == "n":
            return (x1, x2 + env.north)
        elif action == "s":
            return (x1, x2 + env.south)
        elif action == "e":
            return (x1 + env.east, x2)
        elif action == "w":
            return (x1 + env.west, x2)

    def return_counterfactual(original_individual):
        individual = copy.deepcopy(original_individual)
        maxtry = 100        # we need to increase the number of tries for snake dataset, small steps is the reason, increased further for KNN loss
        attempt_no = 0
        path = [original_individual]
        while attempt_no < maxtry:
            individual = torch.from_numpy(individual).type(torch.FloatTensor)
            actions = policy(Variable(individual))
            distribution = Categorical(actions)
            if deter:
                action = torch.argmax(actions)      # replaced with deterministic actions
            else:
                action = distribution.sample()
            new_pt, reward, done = env.step(action.item())
            # action = env.reverse_action_map[action.item()]
            # new_pt = np.array(take_action(*individual, action, env))
            attempt_no += 1
            path.append(new_pt)
            if env.prediction(new_pt.reshape(1, -1)):       # if this is equal to 1
                break
            else:
                individual = new_pt

        print(original_individual, f"successful: {new_pt}",  attempt_no)
        return attempt_no, env.distance_from_manifold (new_pt), path

    successful_transitions = 0
    total_cost = 0
    knn_dist = 0
    trajectories = []
    test_points = 10
    for _ in range(test_points):
        individual = env.reset()
        cost, single_knn_dist, path = return_counterfactual(individual)
        trajectories.append(path)

    print(test_points, "hello")
    trajectories = random.choices(trajectories, k=5)
    plot_trajectories(trajectories, deter, env.dist_lambda)


def learn(n_actions, dist_lambda, episodes, gamma, learning_rate, deter):
    env = environment(n_actions=n_actions, dist_lambda=dist_lambda)

    policy = Policy(env, gamma)
    optimizer = optim.Adam(policy.parameters(), lr=learning_rate)
    final_policy = main(episodes, env, policy, optimizer)
    print(final_policy, "Done")
    use_policy(final_policy, env, deter)


learning_rate = 0.01
gamma = 0.99
dist_lambda = 1.0
n_actions = 4       # in the snake dataset we still have 4 actions, but they are north, south, east, west - with small magnitudes. North, south - 0.05, East, west - 0.1
episodes = 100
deter = False

learn(n_actions, dist_lambda, episodes, gamma, learning_rate, deter)
