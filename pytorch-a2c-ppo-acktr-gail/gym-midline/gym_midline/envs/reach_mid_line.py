import gym, torch
import numpy as np
import random
from gym.utils import seeding


class ReachMidLine(gym.Env):
    metadata = {'render.modes': ['human']}
    """A custom OpenAI gym for toy problem1 - reaching midline and staying there."""
    def __init__(self):
        super(ReachMidLine, self).__init__()
        # 2 discrete actions -- going east or west.
        # self.action_space = gym.spaces.Discrete(2)
        # continuous actions. 
        self.action_space = gym.spaces.Box(low=np.array([-1.0]), high=np.array([1.0]), dtype=np.float)
        # X-axis goes from 4 to 6 and Y-axis goes from 0 to 1.
        self.observation_space = gym.spaces.Box(low=np.array([4.0, 0.0]), high=np.array([6.0, 1.0]), dtype=np.float)
        self.state = None
        self.actions = ["e", "w"]     # "east (0) increases x coordinate, west (1) decreases x coor."
        self.reset()

    # you can restrict movement inside the box or not -- trying not to initially. 
    def step(self, action):
        # import ipdb; ipdb.set_trace()  
        if isinstance(action, torch.Tensor):
            action = action.numpy()[0][0]
            assert isinstance(action, (int, np.int64))
            type_ = 1
        elif isinstance(action, np.ndarray):
            action = action[0]
            type_ = 2
        else:
            raise NotImplementedError
        # print(f"Original: {self.state}")
        if type_ == 1:
            change = 0.05 if action == 0 else (-0.05 if action == 1 else 0.0)
            self.state[0] += change
        elif type_ == 2:
            self.state[0] += action

        distance_from_mid = abs(self.state[0] - 5.0)
        if distance_from_mid >= 1:
            reward = -distance_from_mid**2
        else:    # if less than 1, then squaring has problems. Don't multiply by 2, otherwise moving toward the midline might be more harmful at 1 distance.
            reward = -distance_from_mid
        done = False

        if distance_from_mid < 0.05:
            # done = True       # done = True causes reset in the monitor.py file, therefore turning it off in evaluation. 
            # print(self.state, distance_from_mid, "SEE")
            reward = 5
        info = {}
        # print(f"Hello, {self.state}, {change}, {reward}, {distance_from_mid}")
        return self.state, reward, done, info
    
    def reset(self):
        self.state = self.observation_space.sample()
        return self.state
    
    def render(self, mode='human', close=False):
        print("Here I am: ", self.state)


if __name__ == "__main__":
    x = ReachMidLine()
    import ipdb; ipdb.set_trace()
    print(x.observation_space.sample())
    print(x.state)