import pandas as pd
import numpy as np

class RL(object):
    def __init__(self, actions_space, learning_rate = 0.01, reward_decay = 0.9, e_greedy = 0.9):
        self.actions = actions_space
        self.lr = learning_rate
        self.gamma = reward_decay
        self.epsilon = e_greedy

        self.q_table = pd.DataFrame(columns=self.actions)
        
    def check_state_exist(self, state):
        if state not in self.q_table.index:
            self.q_table = self.q_table.append(
                pd.Series(
                    [0]*len(self.actions),
                    index=self.q_table.columns,
                    name=state,
                )
            )
        
    def choose_action(self, observation):
        self.check_state_exist(observation)
        # action selection
        if np.random.uniform() < self.epsilon:
            # choose best action
            state_action = self.q_table.loc[observation, :]
            # 打乱所有action的位置， good way！
            action = np.random.choice(state_action[state_action == np.max(state_action)].index)
            # print(state_action[state_action == np.max(state_action)].index)
        else:
            action = np.random.choice(self.actions)
        return action

    def learn(self, *args):
        pass


# off-policy
class QLearningTable(RL):
    def __init__(self, actions, learning_rate = 0.01, reward_decay = 0.9, e_greedy = 0.9):
        super(QLearningTable, self).__init__(actions, learning_rate, reward_decay, e_greedy)

    def learn(self, s, a, r, s_):
        self.check_state_exist(s_)
        q_predict = self.q_table.ix[s, a]
        if s_ != 'terminal':
            q_target = r + self.gamma * self.q_table.ix[s_, :].max()
        else:
            q_target = r
        self.q_table[s, a] += self.lr * (q_target - q_predict)


# on-policy
class SarsaTable(RL):
    def __init__(self, actions, learning_rate = 0.01, reward_decay = 0.9, e_greedy = 0.9):
        super(SarsaTable, self).__init__(actions, learning_rate, reward_decay, e_greedy)
        # 让SarsaTable继承自self即（RL），然后用父类的构造函数，所以只需要把一些参数传过去即可

    def learn(self, s, a, r, s_, a_):
        self.check_state_exist(s_)
        q_predict = self.q_table.loc[s, a]
        if s_ != 'terminal':
            q_target = r + self.gamma * self.q_table.loc[s_, a_]
        else:
            q_target = r
        self.q_table.loc[s, a] += self.lr * (q_target - q_predict)

