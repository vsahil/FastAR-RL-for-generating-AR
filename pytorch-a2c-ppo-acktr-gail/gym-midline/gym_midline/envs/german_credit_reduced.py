import gym, torch
import numpy as np
import random, itertools, copy, sys, os
from gym.utils import seeding
import matplotlib.pyplot as plt
sys.path.append("/scratch/vsahil/RL-for-Counterfactuals/paper_expts/")
# from scipy.optimize import fsolve
import pandas as pd
from sklearn.neighbors import NearestNeighbors
import classifier_german as classifier


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


class GermanCreditReduced(gym.Env):
    metadata = {'render.modes': ['human']}
    """ A custom OpenAI gym for the reduced version of German Credit dataset """
    def __init__(self, dist_lambda):
        super(GermanCreditReduced, self).__init__()

        file1 = "/scratch/vsahil/RL-for-Counterfactuals/paper_expts/german_redone.csv"    # 4 is also good
        clf, dataset, scaler, _, _ = classifier.train_model(file=file1, parameter=1)

        # For discrete states and actions
        # file2 = "/scratch/vsahil/RL-for-Counterfactuals/paper_expts/german_redone_4.csv"    # 4 is also good
        # dataset = pd.read_csv(file2)
        # # y = dataset['target']
        # drop_ = ['target','Purpose','Other-debtors','Other-installment-plans','Housing','Telephone','Present-employment-since','Present-residence-since','Property','Savings-account','Number-of-existing-credits','Insatllment-rate','Foreign-worker','Checking-account']
        # dataset = dataset.drop(columns=[*drop_])
        # Discrete action and state space
        self.action_space = gym.spaces.Discrete(2 * len(dataset.columns))
        # self.observation_space = gym.spaces.Discrete(np.prod([len(dataset[i].unique()) for i in dataset.columns]))
        low = np.ones(shape=len(dataset.columns)) * -1.0
        high = np.ones(shape=len(dataset.columns))
        self.observation_space = gym.spaces.Box(low=low, high=high, dtype=np.float)
        self.state = None
        self.dist_lambda = dist_lambda
        self.immutable_features = ['Personal-status', 'Number-of-people-being-lible']
        # self.immutable_features = []
        self.dataset = dataset
        self.state_count = 0
        self.scaler = scaler
        self.classifier = clf
        self.seq = -1
        self.states = {}
        self.states_reverse = {}
        self.no_neighbours = 1
        self.knn_lambda = dist_lambda
        self.knn = NearestNeighbors(n_neighbors=5, p=1)		# 1 would be self, L1 distance makes sense for after normalization. 
        self.knn.fit(scaler.transform(self.dataset))
        # self.knn.fit(self.dataset)
        os.environ['SEQ'] = "-1"
        self.undesirable_x = []
        # env_ = eval_envs.venv.venv.envs[0].env
        for no, i in enumerate(self.dataset.to_numpy()):
            if classifier.predict_single(i, self.scaler, self.classifier) == 0: # and i.tolist() == [1, 3, 0, 3, 4, 1]:    # [0, 3, 0, 2, 4, 1]: # [1, 3, 0, 3, 4, 1]:
                self.undesirable_x.append(tuple(i))
        print(len(self.undesirable_x), "Total points to run the approach on")
        # self.state_sequence()
        self.reset()

    def model(self):
        # for discrete state
        # arr = self.scaler.transform(self.state.reshape(1, -1))
        # probability_class1 = self.classifier.predict_proba(arr.reshape(1,-1))[0][1]	    # find the probability of belonging to class 1 - 

        # for continuous state
        probability_class1 = self.classifier.predict_proba(self.state.reshape(1,-1))[0][1]	    # find the probability of belonging to class 1 - 
        if probability_class1 >= 0.5:
            try:
                assert classifier.predict_single(self.state, self.scaler, self.classifier, pass_scaler=False) == 1
            except:
                import ipdb; ipdb.set_trace()
            return 10, True		# if it is already in good state then very high reward, this should help us get 100% success rate hopefully
        return probability_class1, False

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

        elif isinstance(action, (int, np.int64)):
            type_ = 1

        else:
            raise NotImplementedError
        # print(f"Original: {self.state}")
        info = {}

        if type_ == 1:
            amount = 0.05
            feature_changing = action // 2		# this is the feature that is changing
            decrease = bool(action % 2)
            # import ipdb; ipdb.set_trace()
            reward = -10
            done = False
            # if self.dataset.iloc[:, feature_changing].name in self.immutable_features:
            #     return self.state, reward, done, info

            # age can't decrease
            # elif self.dataset.iloc[:, feature_changing].name == 'age' and decrease:
            #     return self.state, reward, done, info

            # # Job can't decrease
            # elif self.dataset.iloc[:, feature_changing].name == 'Job' and decrease:
            #     return self.state, reward, done, info

            action_ = -amount if decrease else amount
            next_state = list(copy.deepcopy(self.state))
            next_state[feature_changing] = self.state[feature_changing] + action_
            # values = sorted(self.dataset.iloc[:, feature_changing].unique())			# acces column feature changing
            # knn_dist_loss = self.knn_lambda * self.distance_to_closest_k_points(next_state)
            knn_dist_loss = 0
            assert (knn_dist_loss >= 0)

            if decrease:
                # if next_state[feature_changing] > values[0]:
                if next_state[feature_changing] >= -1.0:    # lowest value for a feature is -1.0
                    self.state = np.array(next_state)
                    reward, done = self.model()
                    reward = reward - 1 - knn_dist_loss	    # constant cost for each action		# - self.dist_lambda * self.distance_to_closest_k_points(a+1, b, c)
                else:
                    reward = -10
                    done = False
            else:
                # if next_state[feature_changing] < values[-1]:
                if next_state[feature_changing] <= 1.0:     # highest value possible
                    self.state = np.array(next_state)       # change self.state only if next_state is valid
                    reward, done = self.model()
                    reward = reward - 1 - knn_dist_loss
                else:
                    reward = -10
                    done = False

        elif type_ == 2:
            raise NotImplementedError
            distance = np.clip(action[0], self.action_space.low[0], self.action_space.high[0])
            theta = np.clip(action[1], self.action_space.low[1], self.action_space.high[1])
            self.state[0] += distance * np.cos(theta)
            self.state[1] += distance * np.sin(theta)

        # reward, done = self.total_reward()
        return self.state, reward, done, info

    def distance_to_closest_k_points(self, state):
        nearest_dist, nearest_points = self.knn.kneighbors(np.array([state]).reshape(1,-1), self.no_neighbours, return_distance=True)		# we will take the 5 closest points. We don't need 6 here because the input points are not training pts.
        quantity = np.mean(nearest_dist)		# Now we have that lambda, so no need to divide by self.no_points 
        # print(quantity, nearest_dist)
        return quantity

    def total_reward(self):
        manifold_dist = self.distance_from_manifold()
        classifier_dist = self.distance_from_classifier()
        reward = self.dist_lambda * manifold_dist + classifier_dist - 1       # constant negative reward for taking any action. 
        assert reward <= -1     # always less than -1
        done = False
        # if self.state[0] >= 5.0:
        if -classifier_dist < 0.10:     # remember classifier_dist is negative
            done = True       # done = True causes reset in the monitor.py file, therefore turning it off in evaluation. 
            reward += 5
        return reward, done

    def distance_from_manifold(self):
        # we need to find the perpendicular distance to the closest line. Not all distances otherwise a point on one line will also be penalized
        # import ipdb; ipdb.set_trace()
        point = self.state
        # import ipdb; ipdb.set_trace()
        if -0.5 <= point[0]:    # and point[0] <= 10.0:
            if point[1] > -0.1 and point[1] < 1.0:     # this is the value of np.sin(-0.5/3) which is the lowest point in the curve. 
                dist = self.perpendicular_dist()
                # y_dist = abs(np.sin(point[0]/3) - point[1])
                # x_dist = self.find_x_dist()
            elif point[1] >= 1.0 or point[1] <= -0.1:     # can be replaced by else
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
        # dist = abs(self.state[0] - 5)           # distance from line x = 5
        dist = np.sqrt( (self.state[0] - 5.0)**2 + (self.state[1] - 0.995)**2 )       # the second term is np.sin(5/3)
        # distance = (dist*10)**2      # squaring after multiplying by 10, very important
        return -(dist)      # very negative reward for going far. 
        # return np.sqrt(np.sum(point**2))         # distance from (0, 0), that is our manifold

    def state_sequence(self):
        x = [self.dataset[i].unique() for i in self.dataset.columns]		# for all possible values in the dataset 
        for sts in list(itertools.product(*x)):
            self.states[sts] = self.state_count
            self.states_reverse[self.state_count] = sts
            self.state_count += 1
        assert self.state_count == self.observation_space.n

    # def reset1(self):
    #     # Discrete state and action
    #     # import ipdb; ipdb.set_trace()
    #     self.state = self.observation_space.sample() # random.randint(a=0, b=self.state_count)
    #     # self.state = np.array(self.states_reverse[self.seq])
    #     return self.state

    def reset(self):
        seq = int(os.environ['SEQ'])
        # print("SEQ: ", seq)
        if len(self.undesirable_x) == 0:
            return
        if seq == -1:
            self.state = self.observation_space.sample() # random.randint(a=0, b=self.state_count)
        else:
            # import ipdb; ipdb.set_trace()
            # self.state = self.observation_space.sample()
            self.state = self.scaler.transform(np.array(self.undesirable_x[seq]).reshape(1, -1))[0]
        return self.state

    def render(self, mode='human', close=False):
        print("Here I am: ", self.state)


class GermanCreditReduced01(GermanCreditReduced):
    def __init__(self, enable_render=True):
        super(GermanCreditReduced01, self).__init__(dist_lambda=0.1)


class GermanCreditReduced1(GermanCreditReduced):
    def __init__(self, enable_render=True):
        super(GermanCreditReduced1, self).__init__(dist_lambda=1.0)


class GermanCreditReduced10(GermanCreditReduced):
    def __init__(self, enable_render=True):
        super(GermanCreditReduced10, self).__init__(dist_lambda=10.0)


class GermanCreditReduced100(GermanCreditReduced):
    def __init__(self, enable_render=True):
        super(GermanCreditReduced100, self).__init__(dist_lambda=100.0)


class GermanCreditReduced1000(GermanCreditReduced):
    def __init__(self, enable_render=True):
        super(GermanCreditReduced1000, self).__init__(dist_lambda=1000.0)


if __name__ == "__main__":
    import sys, sklearn
    import pandas as pd
    # from sklearn.model_selection import train_test_split
    sys.path.append("/scratch/vsahil/RL-for-Counterfactuals/paper_expts/")
    import classifier
    file1 = "/scratch/vsahil/RL-for-Counterfactuals/paper_expts/german_redone.csv"    # 4 is also good
    clf, dataset, scaler, X_test, X_train = classifier.train_model(file=file1, parameter=1)

    # For discrete states and actions

    # For continuous states and discrete actions
    file2 = "/scratch/vsahil/RL-for-Counterfactuals/paper_expts/german_redone.csv"    # 4 is also good
    dataset = pd.read_csv(file2)
    y = dataset['target']
    drop_ = ['target','Purpose','Other-debtors','Other-installment-plans','Housing','Telephone','Present-employment-since','Present-residence-since','Property','Savings-account','Number-of-existing-credits','Insatllment-rate','Foreign-worker','Checking-account']
    X = dataset.drop(columns=[*drop_])
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=5)
    x = GermanCreditReduced01()

    # X_train_ = scaler.transform(X_train)
    # X_test_ = scaler.transform(X_test)
    # import ipdb; ipdb.set_trace()
    print(x.state)
    print(x.step(5))
    print(x.observation_space.shape)
    print(x.step(7))
    print(x.step(9))
    print(x.step(1))
    print(x.step(13))
    # import ipdb; ipdb.set_trace()
    print(x.observation_space.sample())
    print(x.state)
