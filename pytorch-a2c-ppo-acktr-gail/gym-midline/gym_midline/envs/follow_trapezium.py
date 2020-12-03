import gym, torch
import numpy as np
import random
from gym.utils import seeding
import matplotlib.pyplot as plt


def plot_curve(env=None):
    plt.figure()
    xs = [0, 2, 8, 10]
    ys = [0, 2, 2, 0]
    plt.ylim(0, 5)
    plt.figure()
    plt.plot(xs, ys)
    if env:
        for points in range(20):
            state = env.reset()
            plt.plot([state[0]], [state[1]], marker='o', markersize=3, color="red")
    plt.savefig("trapezium_curve.png")
    exit(0)


class FollowTrapezium(gym.Env):
    metadata = {'render.modes': ['human']}
    """A custom OpenAI gym for toy problem3 - following sine"""
    def __init__(self, dist_lambda):
        super(FollowTrapezium, self).__init__()
        # 4 discrete actions -- going east, west, north, or south.
        self.action_space = gym.spaces.Discrete(4)
        # X-axis goes from 0 to 7 and Y-axis goes from 0 to 2.
        self.observation_space = gym.spaces.Box(low=np.array([0, 0]), high=np.array([10.0, 3.0]), dtype=np.float)
        self.state = None
        self.actions = ["e", "nw", "w", "nw"]     # "east (0) increases x coordinate, west (1) decreases x coor, north (2) increase y coor, south (3) decreases y coor."
        # self.actions = ["e", "ne"]
        self.y_height = 2.0
        self.x0_boundary = 0.0
        self.x1_boundary = 2.0
        self.x2_boundary = 8.0
        self.x3_boundary = 10.0
        self.dist_lambda = dist_lambda
        self.sine_points_stored = []
        self.reset()

    def step(self, action):
        # print(action, type(action), "see")
        if isinstance(action, torch.Tensor):
            action = action.numpy()[0][0]
        assert isinstance(action, (int, np.int64))
        # print(f"Original: {self.state}")
        amount = 0.05
        if action == 0:     # east
            self.state[0] += amount
        elif action == 1:      # north east
            self.state[0] += amount
            self.state[1] += amount
        elif action == 2:      # west
            self.state[0] -= amount
        elif action == 3:      # north west
            self.state[0] -= amount
            self.state[1] += amount
        reward, done = self.total_reward()
        info = {}
        # print(f"Hello, {self.state}, {change}, {reward}, {distance_from_mid}")
        return self.state, reward, done, info

    def total_reward(self):
        manifold_dist = self.distance_from_manifold()
        classifier_dist = self.distance_from_classifier()
        reward = self.dist_lambda * manifold_dist + classifier_dist - 1       # constant negative reward for taking any action. 
        assert reward <= -1     # always less than -1
        distance_from_mid = abs(self.state[0] - 5)
        done = False
        if distance_from_mid < 0.10:
            done = True
            reward += 5
        return reward, done

    def distance_from_manifold(self):
        # we need to find the perpendicular distance to the closest line.
        point = self.state
        if point[0] <= 2.0:
            dist = abs(point[1] - point[0]) / 1.414  # sqrt(2)
        elif point[0] <= 8.0:
            dist = abs(point[1] - 2)
        else:
            dist = abs(point[1] + point[0] - 10) / 1.414  # sqrt(2)

        return -(dist)

    def distance_from_classifier(self):
        # This is very problematic as for distances less than 1, this will scale down a lot.
        dist = abs(self.state[0] - 5)         # distance from line x = 5
        # distance = (dist*10)**2      # squaring after multiplying by 10, very important
        return -(dist)      # very negative reward for going far. 
        # return np.sqrt(np.sum(point**2))         # distance from (0, 0), that is our manifold

    # Start on the trapezium.
    def reset(self):
        # part = random.randint(1, 3)
        part = random.randint(1, 2)
        if part == 1:
            x_sample = (self.x1_boundary - self.x0_boundary) * np.random.random_sample() + self.x0_boundary
            y_sample = x_sample     # since it is 45 degree line

        elif part == 2:
            # x_sample = (self.x2_boundary - self.x1_boundary) * np.random.random_sample() + self.x1_boundary
            x_sample = (5.0 - self.x1_boundary) * np.random.random_sample() + self.x1_boundary
            y_sample = self.y_height

        elif part == 3:
            x_sample = (self.x3_boundary - self.x2_boundary) * np.random.random_sample() + self.x2_boundary
            y_sample = 10 - x_sample        # this is the equation of the line

        else:
            raise NotImplementedError

        self.state = np.array([x_sample, y_sample])
        return self.state

    def render(self, mode='human', close=False):
        print("Here I am: ", self.state)


class FollowTrapezium01(FollowTrapezium):
    def __init__(self, enable_render=True):
        super(FollowTrapezium01, self).__init__(dist_lambda=0.1)


class FollowTrapezium1(FollowTrapezium):
    def __init__(self, enable_render=True):
        super(FollowTrapezium1, self).__init__(dist_lambda=1.0)


class FollowTrapezium10(FollowTrapezium):
    def __init__(self, enable_render=True):
        super(FollowTrapezium10, self).__init__(dist_lambda=10.0)


class FollowTrapezium100(FollowTrapezium):
    def __init__(self, enable_render=True):
        super(FollowTrapezium100, self).__init__(dist_lambda=100.0)


class FollowTrapezium1000(FollowTrapezium):
    def __init__(self, enable_render=True):
        super(FollowTrapezium1000, self).__init__(dist_lambda=1000.0)


if __name__ == "__main__":
    x = FollowTrapezium1()
    # plot_curve()
    print(x.state)
    print(x.step(1))
    # import ipdb; ipdb.set_trace()
    print(x.observation_space.sample())
    print(x.state)
