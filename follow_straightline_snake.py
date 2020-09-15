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
random.seed(4)


class environment():
    def __init__(self, n_actions, dist_lambda=0):
        self.observation_space = np.zeros([2])
        self.action_space = n_actions
        self.previous_state = np.zeros([2])
        self.current_state = np.zeros([2])
        self.dist_lambda = dist_lambda
        # self.reverse_action_map = {0: 'n', 1: 's', 2: 'e', 3: 'w'}
        # self.action_map = {'n': 0, 's': 1, 'e': 2, 'w': 3}
        self.reverse_action_map = {0: 'e', 1: 'w'}
        self.action_map = {'e': 0, 'w': 1}
        self.min_x1 = 4
        self.max_x1 = 6
        self.north = 0.05
        self.south = -0.05
        self.east = 0.05
        self.west = -0.05

    def step(self, action):
        x1, x2 = self.current_state
        self.previous_state = copy.deepcopy(self.current_state)

        # if self.reverse_action_map[action] == 'n':
        #     self.current_state = np.array([x1, x2 + self.north])
        #     done = self.prediction(self.current_state)

        # elif self.reverse_action_map[action] == 's':
        #     self.current_state = np.array([x1, x2 + self.south])
        #     done = self.prediction(self.current_state)

        if self.reverse_action_map[action] == 'e':
            self.current_state = np.array([x1 + self.east, x2])
            done = self.prediction(self.current_state)

        elif self.reverse_action_map[action] == 'w':
            self.current_state = np.array([x1 + self.west, x2])
            done = self.prediction(self.current_state)

        if int(self.dist_lambda) == 2:
            raise NotImplementedError
            reward = -(self.distance_from_manifold(self.current_state) - self.distance_from_manifold(self.previous_state))     # - distance from manifold
        elif int(self.dist_lambda) == 1:
            reward = self.distance_from_manifold(self.current_state) - 1       # constant negative reward for taking any action. 
        assert reward <= -1     # always less than -1
        return self.current_state, reward, done

    def distance_from_manifold(self, point):
        dist = abs(point[0] - 5)**2         # distance from line x = 5
        # if dist < 1e-2:
        #     return 100
        # else:
        return -(dist**2)      # very negative reward for going far. 
        # return np.sqrt(np.sum(point**2))         # distance from (0, 0), that is our manifold

    def reset(self, set_state=True):
        # maybe starting on the sine curve causes problems. 
        x1_sample = (self.max_x1 - self.min_x1) * np.random.random_sample() + self.min_x1
        assert self.min_x1 <= x1_sample <= self.max_x1
        # this was for (0, 0) manifold
        k = 1
        x2_sample = k * np.random.random_sample()       # samples between 0 and 4
        # x2_sample = np.sin(x1_sample / 3)       # start on the sine curve
        assert 0 <= x2_sample <= k
        if set_state:
            self.current_state = np.array([x1_sample, x2_sample])
        # return np.array([0, 0])
        return np.array([x1_sample, x2_sample])

    def prediction(self, point):
        # if abs(point[0] - 5) < 0.05:
        #     return True
        return False            # let this be false always


class Policy(nn.Module):
    def __init__(self, env, gamma):
        super(Policy, self).__init__()
        self.state_space = env.observation_space.shape[0]
        self.action_space = env.action_space

        self.l1 = nn.Linear(self.state_space, 2, bias=False)
        self.l2 = nn.Linear(2, self.action_space, bias=False)

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


class Policy_linear(nn.Module):

    def __init__(self, env, gamma):
        super(Policy_linear, self).__init__()
        self.state_space = env.observation_space.shape[0]
        self.action_space = env.action_space
        self.linear = torch.nn.Linear(self.state_space, self.action_space)
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
            self.linear,
            # nn.Dropout(p=0.6),
            # nn.ReLU(),
            # self.l2,
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


def main(args, env, policy, optimizer):
    running_reward = 10
    avg_rewards = []
    for episode in range(args.episodes):
        state = env.reset()  # Reset environment and record the starting state
        done = False
        avg_reward = 0
        for time in range(args.max_time):        # 100 is enough for synthetic2.csv, 1000 for snake dataset.
            action = select_action(policy, state)
            # Step through environment using chosen action
            state, reward, done = env.step(action.item())

            # Save reward
            policy.reward_episode.append(reward)
            avg_reward += reward
            if done:
                break

        avg_reward /= (time + 1)
        avg_rewards.append(avg_reward)
        # Used to determine when the environment is solved.
        running_reward = (running_reward * 0.99) + (time * 0.01)
        # if episode >= 1:
        #     plt.figure()
        #     plt.plot(range(episode + 1), avg_rewards)
        #     plt.savefig("avg_reward.png")

        policy = update_policy(policy, optimizer)      # batch size = 1

        # if episode % 10 == 0:
        #     use_policy(policy, env, args)
        print(f'Episode {episode}\tLast length: {time:5d}\tAverage length: {running_reward:.2f}')
    plt.figure()
    plt.plot(range(args.episodes), avg_rewards)
    plt.xlabel('Episodes')
    plt.ylabel('Avg. Reward in an episode')
    plt.axis('tight')
    with open(f"{args.fig_directory}/reward.txt", "a") as f:
        print(args.lr, args.gamma, args.episodes, args.max_time, avg_reward, file=f)
    plt.savefig(f'{args.fig_directory}/avg_reward_{args.lr}_{args.gamma}_{args.episodes}_{args.max_time}.png')
    return policy


def plot_trajectories(x, env, args):
    plt.figure()
    plt.axvline(x=5)
    for path in x:
        xs = [i[0] for i in path]
        ys = [i[1] for i in path]
        markers_on = [0]        # the first and last point ais marked with a shape
        plt.plot(xs, ys, '-D', markevery=markers_on)
        markers_on = [-1]
        plt.plot(xs, ys, '-o', markevery=markers_on)  # the end might be circular
        print(len(path), "see length of trajectory")
        print(env.distance_from_manifold(path[0]), env.distance_from_manifold(path[-1]), "see distances")

    plt.xlabel('x1')
    plt.ylabel('y')
    plt.axis('tight')
    plt.title(f'Reward type = {env.dist_lambda}')
    if args.deter:
        plt.savefig(f'{args.fig_directory}/trajectories_l2_deter_{args.lr}_{args.gamma}_{args.episodes}_{args.max_time}.png')
    else:
        plt.savefig(f'{args.fig_directory}/trajectories_l2_nondeter_{args.lr}_{args.gamma}_{args.episodes}_{args.max_time}.png')
    print("PLOT DONE")


def use_policy(policy, env, args):

    def return_counterfactual(original_individual):
        individual = copy.deepcopy(original_individual)
        maxtry = 100        # we need to increase the number of tries for snake dataset, small steps is the reason, increased further for KNN loss
        attempt_no = 0
        path = [original_individual]
        while attempt_no < maxtry:
            individual = torch.from_numpy(individual).type(torch.FloatTensor)
            actions = policy(Variable(individual))
            distribution = Categorical(actions)
            if args.deter:
                action = torch.argmax(actions)      # replaced with deterministic actions
            else:
                action = distribution.sample()
            new_pt, reward, done = env.step(action.item())
            # action = env.reverse_action_map[action.item()]
            # new_pt = np.array(take_action(*individual, action, env))
            attempt_no += 1
            path.append(new_pt)
            if env.prediction(new_pt):       # if this is equal to 1
                break
            else:
                individual = new_pt

        print(original_individual, f"successful: {new_pt}",  attempt_no)
        return attempt_no, env.distance_from_manifold(new_pt), path

    trajectories = []
    test_points = 20
    for _ in range(test_points):
        individual = env.reset()
        cost, single_knn_dist, path = return_counterfactual(individual)
        trajectories.append(path)

    # print(test_points, "hello")
    trajectories = random.choices(trajectories, k=8)
    plot_trajectories(trajectories, env, args)


def learn(n_actions, dist_lambda, args):
    env = environment(n_actions=n_actions, dist_lambda=dist_lambda)

    # policy = Policy(env, args.gamma)
    policy = Policy_linear(env, args.gamma)
    optimizer = optim.Adam(policy.parameters(), lr=args.lr)
    final_policy = main(args, env, policy, optimizer)
    use_policy(final_policy, env, args)


import argparse

parser = argparse.ArgumentParser(description='run this file')
parser.add_argument('--lr', type=float, default=0.01,
                    help='Learning Rate')
parser.add_argument('--gamma', type=float, default=0.99,
                    help='Discount factor Gamma')
parser.add_argument('--episodes', type=int, default=100,
                    help='No. of episodes')
parser.add_argument('--max_time', type=int, default=1000,
                    help='Time horizon for an episode')
parser.add_argument('--deter', type=int, default=1,
                    help='Deterministic or non-deterministic when in use')
parser.add_argument('--fig_directory', type=str, required=True,
                    help='Directory for storing the figures')
args = parser.parse_args()
args.deter = bool(args.deter)       # convert to boolean
# fig_directory = 'plots/try_follow_straightline1'
# print(args.lr)
# print(args.gamma)
# print(args.episodes)
# print(args.max_time)
# print(args.deter)
# print(args.fig_directory)
# # exit(0)
# import ipdb; ipdb.set_trace()

dist_lambda = 1.0      # lambda = 1.0, when only current distance is used. 
n_actions = 2       # in the snake dataset we still have 4 actions, but they are north, south, east, west - with small magnitudes. North, south - 0.05, East, west - 0.1

learn(n_actions, dist_lambda, args)
# How much effect do hyper-params have, I was seeing complete divergence.
# avg reward per episode 