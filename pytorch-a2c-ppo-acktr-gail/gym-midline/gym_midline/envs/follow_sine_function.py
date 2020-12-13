import gym, torch
import numpy as np
import random
from gym.utils import seeding
import matplotlib.pyplot as plt
from scipy.optimize import fsolve


def plot_curve(env=None):
    plt.figure()
    x1 = np.linspace(-0.5, 10, 201)
    x2 = np.sin(x1 / 3)
    plt.figure()
    plt.plot(x1, x2)
    if env:
        for points in range(20):
            state = env.reset()
            plt.plot([state[0]], [state[1]], marker='o', markersize=3, color="red")
    plt.savefig("sine_curve.png")
    exit(0)


class FollowSine(gym.Env):
    metadata = {'render.modes': ['human']}
    """A custom OpenAI gym for toy problem3 - following sine"""
    def __init__(self, dist_lambda):
        super(FollowSine, self).__init__()
        # 4 discrete actions -- going east, west, north, or south.
        # self.action_space = gym.spaces.Discrete(4)
        # self.action_space = gym.spaces.Discrete(2)
        self.min_action = -0.05      # not set to 0, every action should do something. 
        self.max_action = 0.05
        self.min_theta = 0.0
        self.max_theta = 90.0
        self.action_space = gym.spaces.Box(low=np.array([self.min_action, self.min_theta]), high=np.array([self.max_action, self.max_theta]), dtype=np.float)
        # X-axis goes from 0 to 7 and Y-axis goes from 0 to 2.
        self.observation_space = gym.spaces.Box(low=np.array([-0.5, -0.5]), high=np.array([10.0, 1.5]), dtype=np.float)
        # this is in terms of r and \theta. 
        # self.observation_space = gym.spaces.Box(low=np.array([0.0, 0.0]), high=np.array([11.0, 90.0]), dtype=np.float)
        self.state = None
        # self.actions = ["e", "w", "n", "s"]     # "east (0) increases x coordinate, west (1) decreases x coor, north (2) increase y coor, south (3) decreases y coor."
        # self.actions = ["e", "n"]
        # self.y_height = 0.4
        # self.x0_boundary = 1.0
        # self.x1_boundary = 3.0
        # self.x2_boundary = 6.0
        self.dist_lambda = dist_lambda
        self.sine_points_stored = []
        # self.sine_populate()
        self.reset()

    def sine_populate(self):
        x = np.linspace(-0.5, 10, 2001)
        y = np.sin(x / 3)
        self.sine_points_stored.append(x)
        self.sine_points_stored.append(y)

    def step(self, action):
        # print(action, type(action), "see")
        # import ipdb; ipdb.set_trace()
        if isinstance(action, torch.Tensor):
            action = action.numpy()[0][0]
            assert isinstance(action, (int, np.int64))
            type_ = 1
        elif isinstance(action, np.ndarray):
            # action = action[0]
            type_ = 2
        else:
            raise NotImplementedError
        # print(f"Original: {self.state}")

        if type_ == 1:
            amount = 0.05
            if action == 0:
                self.state[0] += amount
            # elif action == 1:
            #     self.state[0] -= amount
            elif action == 1:
                self.state[1] += amount
            # elif action == 3:
            #     self.state[1] -= amount

        elif type_ == 2:
            distance = np.clip(action[0], self.action_space.low[0], self.action_space.high[0])
            theta = np.clip(action[1], self.action_space.low[1], self.action_space.high[1])
            self.state[0] += distance * np.cos(theta)
            self.state[1] += distance * np.sin(theta)

        reward, done = self.total_reward()
        info = {}
        # print(f"Hello, {self.state}, {distance}, {theta}")
        # print(f"Hello, {self.state}, {change}, {reward}, {distance_from_mid}")
        return self.state, reward, done, info

    def total_reward(self):
        manifold_dist = self.distance_from_manifold()
        classifier_dist = self.distance_from_classifier()
        # remove this -1 and rerun.
        reward = self.dist_lambda * manifold_dist + classifier_dist - 0.05      # constant negative reward for taking any action. 
        # assert reward <= -1     # always less than -1
        done = False
        if self.state[0] >= 4.7:
        # if self.state[0] >= 5.0:
        # if -classifier_dist < 0.10:     # remember classifier_dist is negative
            done = True       # done = True causes reset in the monitor.py file, therefore turning it off in evaluation. 
            reward += 1
        return reward, done

    def find_x_dist(self):
        # find two closest points, one on the same side of midpoint and other on the other side. 
        idx = np.argsort(abs(self.sine_points_stored[1] - self.state[1]))[:2]
        dist1 = abs(self.sine_points_stored[0][idx[0]] - self.state[0])
        dist2 = abs(self.sine_points_stored[0][idx[1]] - self.state[0]) 
        return min(dist1, dist2)

    def perpendicular_dist(self):
        #                (x - a)           +         1/3 *(sin(x/3) - b) * cos(x/3)
        func = lambda x: x - self.state[0] + 1/3*((np.sin(x/3) - self.state[1]) * np.cos(x/3))
        sol = fsolve(func, 1)[0]
        closest_point = np.array([sol, np.sin(sol/3)])
        assert closest_point[0] < 5.0
        assert closest_point.shape == self.state.shape
        return np.sqrt(np.sum((closest_point - self.state)**2))     # euclidean dist. from the closest point. 

    def distance_from_manifold(self):
        # we need to find the perpendicular distance to the closest line. Not all distances otherwise a point on one line will also be penalized
        # import ipdb; ipdb.set_trace()
        point = self.state
        # import ipdb; ipdb.set_trace()
        if -0.5 <= point[0]:    # and point[0] <= 10.0:
            if point[1] >= -0.1 and point[1] <= 1.05:     # this is the value of np.sin(-0.5/3) which is the lowest point in the curve. 
                dist = self.perpendicular_dist()
                # y_dist = abs(np.sin(point[0]/3) - point[1])
                # x_dist = self.find_x_dist()
            elif point[1] > 1.05 or point[1] < -0.1:     # can be replaced by else
                dist = 50**2
                # x_dist = 10     # A high value
                # y_dist = abs(np.sin(point[0]/3) - point[1])
            else:
                raise NotImplementedError

        elif point[0] < -0.5: # or point[0] > 10.0:
            dist = 50**2
            # if point[1] > -0.1 and point[1] < 1.0:
            #     x_dist = self.find_x_dist()
            #     y_dist = 10     # A high value
            # else:
            #     x_dist = y_dist = 10

        else:
            raise NotImplementedError

        # assert x_dist >= 0 and y_dist >= 0, (x_dist, y_dist)
        # dist = np.sqrt(x_dist**2 + y_dist**2)      # perpendicular distance. Srqt important when x_dist and y_dist are less than 1. 

        return -(dist)

    def distance_from_classifier(self):
        # This is very problematic as for distances less than 1, this will scale down a lot.
        dist = abs(self.state[0] - 4.7)           # distance from line x = 5
        # dist = np.sqrt( (self.state[0] - 5.0)**2 + (self.state[1] - 0.995)**2 )       # the second term is np.sin(5/3)
        # distance = (dist*10)**2      # squaring after multiplying by 10, very important
        return -(dist)      # very negative reward for going far. 
        # return np.sqrt(np.sum(point**2))         # distance from (0, 0), that is our manifold

    # Start on the left half of the sine curve.
    def reset(self):
        # x_sample, _ = self.observation_space.sample()
        # x_sample = 5.5 * np.random.random_sample() - 0.5
        x_sample = 5.2 * np.random.random_sample() - 0.5
        y_sample = np.sin(x_sample / 3)
        self.state = np.array([x_sample, y_sample])
        return self.state

    def render(self, mode='human', close=False):
        print("Here I am: ", self.state)


class FollowSine01(FollowSine):
    def __init__(self, enable_render=True):
        super(FollowSine01, self).__init__(dist_lambda=0.1)


class FollowSine1(FollowSine):
    def __init__(self, enable_render=True):
        super(FollowSine1, self).__init__(dist_lambda=1.0)


class FollowSine10(FollowSine):
    def __init__(self, enable_render=True):
        super(FollowSine10, self).__init__(dist_lambda=10.0)


class FollowSine100(FollowSine):
    def __init__(self, enable_render=True):
        super(FollowSine100, self).__init__(dist_lambda=100.0)


class FollowSine1000(FollowSine):
    def __init__(self, enable_render=True):
        super(FollowSine1000, self).__init__(dist_lambda=1000.0)


class FollowSine10000(FollowSine):
    def __init__(self, enable_render=True):
        super(FollowSine10000, self).__init__(dist_lambda=10000.0)


if __name__ == "__main__":
    x = FollowSine1()
    # plot_curve()
    print(x.state)
    print(x.step(1))
    # import ipdb; ipdb.set_trace()
    print(x.observation_space.sample())
    print(x.state)
