import gym, torch, time
import numpy as np
import random, itertools, copy, sys, os
from gym.utils import seeding
import matplotlib.pyplot as plt
sys.path.append("/scratch/vsahil/RL-for-Counterfactuals/paper_expts/")
# from scipy.optimize import fsolve
import pandas as pd
from sklearn.neighbors import NearestNeighbors
import classifier_german as classifier


class GermanCredit(gym.Env):
    metadata = {'render.modes': ['human']}
    """ A custom OpenAI gym for the reduced version of German Credit dataset """
    def __init__(self, dist_lambda):
        super(GermanCredit, self).__init__()

        file1 = "/scratch/vsahil/RL-for-Counterfactuals/paper_expts/german_redone.csv"    # 4 is also good
        # file1 = "/scratch/vsahil/RL-for-Counterfactuals/paper_expts/german_onehot.csv"    # 4 is also good
        clf, dataset, scaler, X_test, X_train = classifier.train_model_german(file=file1, parameter=1)
        # Discrete action space
        self.action_space = gym.spaces.Discrete(2 * len(dataset.columns))

        # Continous action space
        # self.min_action = -1
        # self.max_action = 1
        # self.min_feature = -1     # this range will correspond to #len(dataset.columns) features
        # self.max_feature = 0.99
        # self.action_space = gym.spaces.Box(low=np.array([self.min_action, self.min_feature]), high=np.array([self.max_action, self.max_feature]), dtype=np.float)

        low = np.ones(shape=len(dataset.columns)) * -1.0
        high = np.ones(shape=len(dataset.columns))
        self.observation_space = gym.spaces.Box(low=low, high=high, dtype=np.float)
        self.state = None
        self.dist_lambda = dist_lambda
        self.immutable_features = ['Personal-status', 'Number-of-people-being-lible', 'Foreign-worker', 'Purpose']
        self.dataset = dataset
        # self.train_dataset = pd.DataFrame(scaler.transform(X_train), columns=self.dataset.columns)
        self.train_dataset = scaler.transform(X_train)
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
        self.numerical_features = [1, 4, 7, 10, 12, 15, 17]
        # self.knn.fit(self.dataset)
        os.environ['SEQ'] = "-1"
        self.undesirable_x = []
        # env_ = eval_envs.venv.venv.envs[0].env
        try:
            self.undesirable_x = np.load(f"{os.path.dirname(os.path.realpath(__file__))}/../../../../baselines/undesirable_x_german.npy")
            print("Found")
        except:
            print("Not Found")
            # print(f"{os.path.dirname(os.path.realpath(__file__))}/../../../../baselines/undesirable_x.npy")
            assert not os.path.exists(f"{os.path.dirname(os.path.realpath(__file__))}/../../../../baselines/undesirable_x_german.npy")
            undesirable_x = []
            for no, i in enumerate(self.dataset.to_numpy()):
                if self.classifier.predict(self.scaler.transform(i.reshape(1, -1))) == 0:
                # if classifier.predict_single(i, self.scaler, self.classifier) == 0:
                    undesirable_x.append(tuple(i))
            # undesirable_x = undesirable_x
            self.undesirable_x = np.array(undesirable_x)
            np.save(f"{os.path.dirname(os.path.realpath(__file__))}/../../../../baselines/undesirable_x_german.npy", undesirable_x)

        # for no, i in enumerate(self.dataset.to_numpy()):
        #     if classifier.predict_single(i, self.scaler, self.classifier) == 0: # and i.tolist() == [1, 3, 0, 3, 4, 1]:    # [0, 3, 0, 2, 4, 1]: # [1, 3, 0, 3, 4, 1]:
        #         self.undesirable_x.append(tuple(i))
        
        print(len(self.undesirable_x), "Total points to run the approach on")
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
            return 100, True		# if it is already in good state then very high reward, this should help us get 100% success rate hopefully
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
            feature_changing = action // 2		# this is the feature that is changing
            decrease = bool(action % 2)
            if decrease:
                amount = -0.05
            else:
                amount = 0.05

        elif type_ == 2:
            decrease = False
            amount = np.clip(action[0], self.action_space.low[0], self.action_space.high[0])
            if amount < 0:
                decrease = True
            feature = np.clip(action[1], self.action_space.low[1], self.action_space.high[1])
            feature += 1        # casts in 0 to 2 range
            feature_changing = int(feature * (len(self.dataset.columns) // 2))      # we need int not round
        else:
            assert False

        reward = -10
        done = False
        # if self.dataset.iloc[:, feature_changing].name in self.immutable_features:

        # for imf in self.immutable_features:
        #     if imf in self.dataset.iloc[:, feature_changing].name:
        #         return self.state, reward, done, info

        # # age can't decrease
        # if self.dataset.iloc[:, feature_changing].name == 'age' and decrease:
        #     return self.state, reward, done, info

        # # Job can't decrease
        # elif self.dataset.iloc[:, feature_changing].name == 'Job' and decrease:
        #     return self.state, reward, done, info

        action_ = amount
        next_state = list(copy.deepcopy(self.state))
        next_state[feature_changing] = self.state[feature_changing] + action_
        # values = sorted(self.dataset.iloc[:, feature_changing].unique())			# acces column feature changing
        knn_dist_loss = self.knn_lambda * self.distance_to_closest_k_points(next_state)
        # knn_dist_loss = 0
        assert (knn_dist_loss >= 0)
        constant = 0        # constant loss for each action

        if decrease:
            if next_state[feature_changing] > -1.0:    # lowest value for a feature is -1.0
                self.state = np.array(next_state)
                reward, done = self.model()
                reward = reward - constant - knn_dist_loss	    # constant cost for each action		# - self.dist_lambda * self.distance_to_closest_k_points(a+1, b, c)
            else:
                reward = -10
                done = False
        else:
            if next_state[feature_changing] < 1.0:     # highest value possible
                self.state = np.array(next_state)       # change self.state only if next_state is valid
                reward, done = self.model()
                reward = reward - constant - knn_dist_loss
            else:
                reward = -10
                done = False

        # reward, done = self.total_reward()
        return self.state, reward, done, info

    def distance_to_closest_k_points(self, state):
        # import ipdb; ipdb.set_trace()
        nearest_dist, nearest_points = self.knn.kneighbors(np.array([state]).reshape(1,-1), self.no_neighbours, return_distance=True)		# we will take the 5 closest points. We don't need 6 here because the input points are not training pts.
        quantity = np.mean(nearest_dist)		# Now we have that lambda, so no need to divide by self.no_points 
        # print(quantity, nearest_dist)
        return quantity

    def reset(self):
        seq = int(os.environ['SEQ'])
        # print("SEQ: ", seq)
        # import ipdb; ipdb.set_trace()
        if len(self.undesirable_x) == 0:
            return
        if seq == -1:
            # self.state = self.observation_space.sample()    # random.randint(a=0, b=self.state_count)
            # self.state = self.train_dataset.sample().to_numpy().reshape(20,)      # from pandas dataframe
            idx = random.randrange(self.train_dataset.shape[0])
            self.state = self.train_dataset[idx]
        else:
            # This is used during evaluation of a trained agent
            self.state = self.scaler.transform(np.array(self.undesirable_x[seq]).reshape(1, -1))[0]
        return self.state

    def render(self, mode='human', close=False):
        print("Here I am: ", self.state)


class GermanCredit0(GermanCredit):
    def __init__(self, enable_render=True):
        super(GermanCredit0, self).__init__(dist_lambda=0.0)


class GermanCredit01(GermanCredit):
    def __init__(self, enable_render=True):
        super(GermanCredit01, self).__init__(dist_lambda=0.1)


class GermanCredit1(GermanCredit):
    def __init__(self, enable_render=True):
        super(GermanCredit1, self).__init__(dist_lambda=1.0)


class GermanCredit10(GermanCredit):
    def __init__(self, enable_render=True):
        super(GermanCredit10, self).__init__(dist_lambda=10.0)


class GermanCredit100(GermanCredit):
    def __init__(self, enable_render=True):
        super(GermanCredit100, self).__init__(dist_lambda=100.0)


class GermanCredit1000(GermanCredit):
    def __init__(self, enable_render=True):
        super(GermanCredit1000, self).__init__(dist_lambda=1000.0)


if __name__ == "__main__":
    import sys, sklearn
    import pandas as pd
    # from sklearn.model_selection import train_test_split
    sys.path.append("/scratch/vsahil/RL-for-Counterfactuals/paper_expts/")
    import classifier_german as classifier
    # file1 = "/scratch/vsahil/RL-for-Counterfactuals/paper_expts/german_redone.csv"    # 4 is also good
    # clf, dataset, scaler, X_test, X_train = classifier.train_model_german(file=file1, parameter=1)

    # For discrete states and actions

    # For continuous states and discrete actions
    # file2 = "/scratch/vsahil/RL-for-Counterfactuals/paper_expts/german_redone.csv"    # 4 is also good
    # dataset = pd.read_csv(file2)
    # y = dataset['target']
    # drop_ = ['target','Purpose','Other-debtors','Other-installment-plans','Housing','Telephone','Present-employment-since','Present-residence-since','Property','Savings-account','Number-of-existing-credits','Insatllment-rate','Foreign-worker','Checking-account']
    # X = dataset.drop(columns=[*drop_])
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=5)
    st = time.time()
    x = GermanCredit01()
    t1 = time.time()

    # X_train_ = scaler.transform(X_train)
    # X_test_ = scaler.transform(X_test)
    # import ipdb; ipdb.set_trace()
    # print(x.state)
    print(x.step(5))
    # print(x.observation_space.shape)
    print(x.step(7))
    print(x.step(9))
    print(x.step(1))
    print(x.step(13))
    # import ipdb; ipdb.set_trace()
    # print(x.observation_space.sample())
    t2 = time.time()
    # print(x.state)
    print(t1 - st)
    print(t2 - t1)
