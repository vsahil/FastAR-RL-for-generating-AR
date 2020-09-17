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


# the curve will be lines x = 3 [y = 0, 2] then y = 2 [x=3, 6] then x = 6 [y = 0, 2]. 
# Classifier is the line x >= 5. The initial points will be sampled on the left side of the curve. Remember to start with the points on the curve, not random points. 


def plot_curve(env=None):
    plt.figure()
    xs = [1, 3, 3, 6, 6]
    ys = [0, 0, 2, 2, 0]
    plt.xlim(0, 7)
    plt.ylim(0, 4)
    plt.plot(xs, ys)
    if env:
        for points in range(20):
            state = env.reset()
            plt.plot([state[0]], [state[1]], marker='o', markersize=3, color="red")
    plt.savefig("straightline_curve.png")
    exit(0)

# plot_curve()


class environment():
    def __init__(self, n_actions, dist_lambda=0):
        self.observation_space = np.zeros([2])
        self.action_space = n_actions
        self.previous_state = np.zeros([2])
        self.current_state = np.zeros([2])
        self.dist_lambda = dist_lambda
        self.reverse_action_map = {0: 'n', 1: 's', 2: 'e', 3: 'w'}
        self.action_map = {'n': 0, 's': 1, 'e': 2, 'w': 3}
        self.north = 0.05
        self.south = -0.05
        self.east = 0.05
        self.west = -0.05

    def step(self, action):
        x1, x2 = self.current_state
        self.previous_state = copy.deepcopy(self.current_state)

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

        # print(manifold_dist, classifier_dist, self.current_state, "See the rewards")
        return self.current_state, self.total_reward(self.current_state), done

    def total_reward(self, point):
        manifold_dist = self.distance_from_manifold(point)
        classifier_dist = self.distance_from_classifier(point)
        reward = self.dist_lambda * manifold_dist + classifier_dist - 5       # constant negative reward for taking any action. 
        assert reward <= -5     # always less than -1
        return reward
    
    def distance_from_manifold(self, point):
        # we need to find the perpendicular distance to the closest line. Not all distances otherwise a point on one line will also be penalized
        if point[1] >= 0.0 and point[0] <= 3:
            perp1 = point[1]       # distance from part 1
            perp2 = 3.0 - point[0]        # distance from part 2
            assert perp1 >= 0 and perp2 >= 0
            dist = min(perp1, perp2)
            if point[0] < 1.0:
                dist = 50**3          # very very negative reward for going west of x = 1
        
        elif point[1] < 0.0 and point[0] <= 3:
            dist = abs(point[1])

        elif point[1] >= 2.0 and point[0] >= 3:
            dist = point[1] - 2.0
            assert dist >= 0
        
        elif point[1] < 2.0 and point[0] >= 3:
            perp1 = 2.0 - point[1]        # distance from part 3
            perp2 = point[0] - 3.0       # distance from part 2
            assert perp1 >= 0 and perp2 >= 0
            dist = min(perp1, perp2)

        else:
            print(point, "not falls in any region")
            raise NotImplementedError
        
        return -(dist*10)**2

    def distance_from_classifier(self, point):
        # This is very problematic as for distances less than 1, this will scale down a lot.
        dist = abs(point[0] - 5)         # distance from line x = 5
        distance = (dist*10)**2      # squaring after multiplying by 10, very important
        return -(distance)      # very negative reward for going far. 
        # return np.sqrt(np.sum(point**2))         # distance from (0, 0), that is our manifold

    def reset(self, set_state=True):
        # we have 3 sections of the curve, the two line parallel to x, axis and one line parallel to y axis. We should sample equal from them, so use uniform probability.
        part = random.randint(1, 3)
        if part == 1:
            x2_sample = 0
            x1_sample = (3 - 1) * np.random.random_sample() + 1
        
        elif part == 2:
            x1_sample = 3
            x2_sample = 2 * np.random.random_sample()
        
        elif part == 3:
            x2_sample = 2
            x1_sample = (5 - 3) * np.random.random_sample() + 3
        
        if set_state:
            self.current_state = np.array([x1_sample, x2_sample])
        return np.array([x1_sample, x2_sample])

    def prediction(self, point):
        if point[0] >= 5.0:
            return True
        return False


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
    avg_rewards_list = []
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

        avg_reward /= (time + 1)    # this should not be args.max_time because episodes can end earlier. 
        avg_rewards_list.append(round(avg_reward, 2))
        # Used to determine when the environment is solved.
        running_reward = (running_reward * 0.99) + (time * 0.01)
        # if episode >= 1:
        #     plt.figure()
        #     plt.plot(range(episode + 1), avg_rewards_list)
        #     plt.savefig("avg_reward.png")

        policy = update_policy(policy, optimizer)      # batch size = 1

        # if episode % 10 == 0:
        #     _ = use_policy(policy, env, args)
        print(f'Episode {episode}\tLast length: {time:5d}\tAverage length: {running_reward:.2f}')
    plt.figure()
    plt.plot(range(args.episodes), avg_rewards_list)
    plt.xlabel('Episodes')
    plt.ylabel('Avg. Reward in an episode')
    plt.axis('tight')
    with open(f"{args.fig_directory}/reward_episode_{args.lr}_{args.gamma}_{args.dist_lambda}_{args.episodes}_{args.max_time}.txt", "w") as f1:
        for i in avg_rewards_list:
            print(i, file=f1)
    plt.savefig(f'{args.fig_directory}/avg_reward_{args.lr}_{args.gamma}_{args.dist_lambda}_{args.episodes}_{args.max_time}.png')
    # with open(f"{args.fig_directory}/reward.txt", "a") as f:
    #     print(args.lr, args.gamma, args.episodes, args.max_time, avg_reward, file=f)
    return policy, avg_rewards_list


def plot_trajectories_later(x, env, args):
    plt.figure()
    plt.axvline(x=5)
    xs = [1, 3, 3, 6, 6]
    ys = [0, 0, 2, 2, 0]
    plt.xlim(0, 7)
    plt.ylim(0, 5)
    plt.plot(xs, ys)
    plt.xlabel('x1')
    plt.ylabel('y')
    plt.axis('tight')
    plt.title('Reward type = -(10 * absolute_distance)**2')
    import ipdb; ipdb.set_trace()
    for path in x:
        xs = [i[0] for i in path]
        ys = [i[1] for i in path]
        # markers_on = [0]        # the first and last point ais marked with a shape
        # plt.plot(xs, ys, '-D', markevery=markers_on)
        # markers_on = [-1]
        # plt.plot(xs, ys, '-o', markevery=markers_on)  # the end might be circular
        plt.plot(xs, ys, 'bo')
        print(path[0], path[-1], f"see length of trajectory:{len(path)}")
        print(env.distance_from_manifold(path[0]), env.distance_from_manifold(path[-1]), "see distances")
        plt.savefig("trajectory.png")
    # if args.deter:
    #     plt.savefig(f'{args.fig_directory}/trajectories_l2_deter_{args.lr}_{args.gamma}_{args.dist_lambda}_{args.episodes}_{args.max_time}.png')
    # else:
    #     plt.savefig(f'{args.fig_directory}/trajectories_l2_nondeter_{args.lr}_{args.gamma}_{args.dist_lambda}_{args.episodes}_{args.max_time}.png')
    print("PLOT DONE")


def plot_trajectories(x, env, args):
    plt.figure()
    plt.axvline(x=5)
    xs = [1, 3, 3, 6, 6]
    ys = [0, 0, 2, 2, 0]
    plt.xlim(0, 7)
    plt.ylim(0, 5)
    plt.plot(xs, ys)
    for path in x:
        xs = [i[0] for i in path]
        ys = [i[1] for i in path]
        markers_on = [0]        # the first and last point ais marked with a shape
        plt.plot(xs, ys, '-D', markevery=markers_on)
        markers_on = [-1]
        plt.plot(xs, ys, '-o', markevery=markers_on)  # the end might be circular
        print(path[0], path[-1], f"see length of trajectory:{len(path)}")
        print(env.distance_from_manifold(path[0]), env.distance_from_manifold(path[-1]), "see distances")

    plt.xlabel('x1')
    plt.ylabel('y')
    plt.axis('tight')
    plt.title('Reward type = -(10 * absolute_distance)**2')
    if args.deter:
        plt.savefig(f'{args.fig_directory}/trajectories_l2_deter_{args.lr}_{args.gamma}_{args.dist_lambda}_{args.episodes}_{args.max_time}.png')
    else:
        plt.savefig(f'{args.fig_directory}/trajectories_l2_nondeter_{args.lr}_{args.gamma}_{args.dist_lambda}_{args.episodes}_{args.max_time}.png')
    print("PLOT DONE")


def use_policy(policy, env, args, load=False):

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
        return attempt_no, env.total_reward(new_pt), path

    trajectories = []
    test_points = 20
    final_errors = []
    for _ in range(test_points):
        individual = env.reset()
        cost, single_dist, path = return_counterfactual(individual)
        final_errors.append(single_dist)
        trajectories.append((path, single_dist))

    trajectories = random.choices(trajectories, k=12)
    # for i in trajectories:
    #     print(i)
    trajectories = [x[0] for x in trajectories]     # only the paths to be plotted
    if not load:
        plot_trajectories(trajectories, env, args)
    else:
        plot_trajectories_later(trajectories, env, args)
    return final_errors


def learn(n_actions, args, load=False):

    def read_reward_file(filename):
        with open(filename) as f:
            content = f.readlines()
        # you may also want to remove whitespace characters like `\n` at the end of each line
        content = [float(x.strip()) for x in content]
        return content

    env = environment(n_actions=n_actions, dist_lambda=args.dist_lambda)
    # plot_curve(env)
    # import ipdb; ipdb.set_trace()
    # policy = Policy(env, args.gamma)
    policy = Policy_linear(env, args.gamma)
    path = f'{args.fig_directory}/model_{args.lr}_{args.gamma}_{args.episodes}_{args.max_time}.pth'
    if not load:
        optimizer = optim.Adam(policy.parameters(), lr=args.lr)
        final_policy, avg_rewards_list = main(args, env, policy, optimizer)
        torch.save(final_policy.state_dict(), path)
        final_errors = use_policy(final_policy, env, args, load=False)
        # print(final_errors, "see")
        with open(f"{args.fig_directory}/reward.txt", "a") as f:
            print(args.lr, args.gamma, args.dist_lambda, args.episodes, args.max_time, round(np.mean(avg_rewards_list[-50:]), 2), avg_rewards_list[-1], round(np.mean(final_errors), 2), round(np.median(final_errors), 2), file=f)
    else:
        # import ipdb; ipdb.set_trace()
        policy.load_state_dict(torch.load(path))
        final_errors = use_policy(policy, env, args, load=False)
        avg_rewards_list = read_reward_file(f"{args.fig_directory}/reward_episode_{args.lr}_{args.gamma}_{args.dist_lambda}_{args.episodes}_{args.max_time}.txt")
        with open(f"{args.fig_directory}/reward1.txt", "a") as f:
            print(args.lr, args.gamma, args.dist_lambda, args.episodes, args.max_time, round(np.mean(avg_rewards_list[-50:]), 2), avg_rewards_list[-1], round(np.mean(final_errors), 2), round(np.median(final_errors), 2), file=f)
        # print(final_errors)

import argparse

parser = argparse.ArgumentParser(description='run this file')
parser.add_argument('--lr', type=float, default=0.01,
                    help='Learning Rate')
parser.add_argument('--gamma', type=float, default=0.99,
                    help='Discount factor Gamma')
parser.add_argument('--dist_lambda', type=float, default=1.0,
                    help='Multiplying factor with distance')
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

# with open("configs.txt", "a") as f:
#     print(f"Started: {args}", file=f)
    
# fig_directory = 'plots/try_follow_straightline1'
# print(args.lr)
# print(args.gamma)
# print(args.episodes)
# print(args.max_time)
# print(args.deter)
# print(args.fig_directory)
# # exit(0)
# import ipdb; ipdb.set_trace()

dist_lambda = args.dist_lambda      # lambda = 1.0, when only current distance is used. 
n_actions = 4       # in the snake dataset we still have 4 actions, but they are north, south, east, west - with small magnitudes. North, south - 0.05, East, west - 0.1

learn(n_actions, args, load=False)
# How much effect do hyper-params have, I was seeing complete divergence.
# avg reward per episode 