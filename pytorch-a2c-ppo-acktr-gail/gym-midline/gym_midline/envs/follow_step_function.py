import gym, torch
import numpy as np
import random
from gym.utils import seeding
import matplotlib.pyplot as plt


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


class FollowStep(gym.Env):
    metadata = {'render.modes': ['human']}
    """A custom OpenAI gym for toy problem2 - following step"""
    def __init__(self, dist_lambda):
        super(FollowStep, self).__init__()
        # 2 discrete actions -- going east or north.
        # self.action_space = gym.spaces.Discrete(2)
        # 2 continuous actions but only positive values possible. 
        self.min_action = 0.0
        self.max_action = 0.2
        self.action_space = gym.spaces.Box(low=np.array([self.min_action, self.max_action]), high=np.array([self.max_action, self.max_action]), dtype=np.float)
        # X-axis goes from 0 to 7 and Y-axis goes from 0 to 2.
        self.min_x = 0.0
        self.max_x = 7.0
        self.min_y = 0.0
        self.max_y = 2.0
        self.observation_space = gym.spaces.Box(low=np.array([self.min_x, self.min_y]), high=np.array([self.max_x, self.max_y]), dtype=np.float)
        
        self.state = None
        self.actions = ["e", "n"]     # "east (0) increases x coordinate, west (1) decreases x coor."
        self.y_height = 0.4
        self.x0_boundary = 1.0
        self.x1_boundary = 3.0
        self.x2_boundary = 6.0
        self.dist_lambda = dist_lambda
        self.reset()

    def step(self, action):
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
            change = 0 if action == 0 else (1 if action == 1 else NotImplementedError)
            self.state[change] += 0.05      # north and east are both increasing respective coordinates 

        elif type_ == 2:
            a0 = min(max(action[0], self.min_action), self.max_action)
            a1 = min(max(action[1], self.min_action), self.max_action)
            # self.state += action
            self.state[0] += a0
            self.state[1] += a1
            if self.state[0] > self.max_x:
                self.state[0] = self.max_x
            elif self.state[0] < self.min_x:
                self.state[0] = self.min_x

            if self.state[1] > self.max_y:
                self.state[1] = self.max_y
            elif self.state[1] < self.min_y:
                self.state[1] = self.min_y

        reward, done = self.total_reward()
        info = {}
        # print(f"Hello, {self.state}, {change}, {reward}, {distance_from_mid}")
        return self.state, reward, done, info

    def total_reward(self):
        manifold_dist = self.distance_from_manifold()
        classifier_dist = self.distance_from_classifier()
        reward = self.dist_lambda * manifold_dist + classifier_dist - 1       # constant negative reward for taking any action. 
        assert reward <= -1     # always less than -1
        done = False
        if self.state[0] >= 5.0:
            # done = True
            reward += 5
        return reward, done

    def distance_from_manifold(self):
        # we need to find the perpendicular distance to the closest line. Not all distances otherwise a point on one line will also be penalized
        # x2_boundary = 6.0
        point = self.state
        if point[1] >= 0.0 and point[0] <= self.x1_boundary:
            if point[1] <= self.y_height:
                perp1 = point[1]       # distance from part 1
                perp2 = self.x1_boundary - point[0]        # distance from part 2
                assert perp1 >= 0 and perp2 >= 0
                dist = min(perp1, perp2)
            else:
                dist = point[1]     # for point above y height, calculate distance from part 1
            
            if point[0] < self.x0_boundary:
                dist = 50**2          # very very negative reward for going west of x = 1

        elif point[1] < 0.0 and point[0] <= self.x1_boundary:
            dist = abs(point[1])

        elif point[1] >= self.y_height and point[0] >= self.x1_boundary:
            dist = point[1] - self.y_height
            assert dist >= 0

        elif point[1] < self.y_height and point[0] >= self.x1_boundary:
            if point[1] >= 0:
                perp1 = self.y_height - point[1]        # distance from part 3
                perp2 = point[0] - self.x1_boundary       # distance from part 2
                assert perp1 >= 0 and perp2 >= 0
                dist = min(perp1, perp2)
            else:
                dist = self.y_height - point[1]

        else:
            print(point, "not falls in any region")
            raise NotImplementedError

        return -(dist)
        # return -(dist*10)**2

    def distance_from_classifier(self):
        # This is very problematic as for distances less than 1, this will scale down a lot.
        dist = abs(self.state[0] - 5)         # distance from line x = 5
        # distance = (dist*10)**2      # squaring after multiplying by 10, very important
        return -(dist)      # very negative reward for going far. 
        # return np.sqrt(np.sum(point**2))         # distance from (0, 0), that is our manifold

    # This time I want to start on the curve. 
    def reset(self):
        part = random.randint(1, 3)
        if part == 1:
            y_sample = 0
            x_sample = (self.x1_boundary - self.x0_boundary) * np.random.random_sample() + self.x0_boundary
        
        elif part == 2:
            x_sample = self.x1_boundary
            y_sample = self.y_height * np.random.random_sample()

        elif part == 3:
            y_sample = self.y_height
            x_sample = (5 - self.x1_boundary) * np.random.random_sample() + self.x1_boundary
        
        else:
            raise NotImplementedError

        self.state = np.array([x_sample, y_sample])
        # self.state = self.observation_space.sample()
        return self.state

    def render(self, mode='human', close=False):
        print("Here I am: ", self.state)


class FollowStep01(FollowStep):
    def __init__(self, enable_render=True):
        super(FollowStep01, self).__init__(dist_lambda=0.1)


class FollowStep1(FollowStep):
    def __init__(self, enable_render=True):
        super(FollowStep1, self).__init__(dist_lambda=1.0)


class FollowStep10(FollowStep):
    def __init__(self, enable_render=True):
        super(FollowStep10, self).__init__(dist_lambda=10.0)


class FollowStep100(FollowStep):
    def __init__(self, enable_render=True):
        super(FollowStep100, self).__init__(dist_lambda=100.0)


class FollowStep1000(FollowStep):
    def __init__(self, enable_render=True):
        super(FollowStep1000, self).__init__(dist_lambda=1000.0)


if __name__ == "__main__":
    x = FollowStep_1()
    import ipdb; ipdb.set_trace()
    print(x.observation_space.sample())
    print(x.state)
