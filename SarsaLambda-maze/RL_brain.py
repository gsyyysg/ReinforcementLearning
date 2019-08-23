import pandas as pd
import numpy as np


class RL(object):
    def __init__(self, actions_space, learning_rate=0.01, reward_decay=0.9, e_greedy=0.9):
        self.actions = actions_space
        self.lr = learning_rate
        self.gamma = reward_decay
        self.epsilon = e_greedy

        self.q_table = pd.DataFrame(columns=self.actions, dtype=np.float64)

    def check_state_exist(self, state):
        if state not in self.q_table.index:
            self.q_table = self.q_table.append(
                pd.Series(
                    [0] * len(self.actions),
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
    def __init__(self, actions, learning_rate=0.01, reward_decay=0.9, e_greedy=0.9):
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
class SarsaLambdaTable(RL):
    def __init__(self, actions, learning_rate=0.01, reward_decay=0.9, e_greedy=0.9, trace_decay = 0.9):
        super(SarsaLambdaTable, self).__init__(actions, learning_rate, reward_decay, e_greedy)
        # 让SarsaTable继承自self即（RL），然后用父类的构造函数，所以只需要把一些参数传过去即可

        # background view, eligibility trace.
        self.lambda_ = trace_decay  # sarsa-lambda中
        self.eligibility_trace = self.q_table.copy()  # 同样的state-action表，类似q-table

    def check_state_exist(self, state):
        if state not in self.q_table.index:
            to_be_append = pd.Series(
                [0] * len(self.actions),
                index=self.q_table.columns,
                name=state,
            )
            self.q_table = self.q_table.append(to_be_append)

            self.eligibility_trace = self.eligibility_trace.append(to_be_append)

    def learn(self, s, a, r, s_, a_):
        self.check_state_exist(s_)
        q_predict = self.q_table.loc[s, a]
        if s_ != 'terminal':
            q_target = r + self.gamma * self.q_table.loc[s_, a_]
        else:
            q_target = r
        # self.q_table.loc[s, a] += self.lr * (q_target - q_predict)
        error = q_target - q_predict

        # increase trace amount for visited state-action pair

        # Method 1: 每次都增加1， 不可或缺性的值可能会升到很大
        # self.eligibility_trace.loc[s, a]+=1

        # Method 2: 达到封顶
        self.eligibility_trace.loc[s, :] *= 0
        self.eligibility_trace.loc[s, a] = 1

        # Q update， key point
        self.q_table += self.lr * error * self.eligibility_trace

        # decay eligibility trace after update
        # 衰变的体现
        # 这里的含义是表格中每一个位置都乘以衰变值，实现最后结果的相关度是不同的
        self.eligibility_trace *= self.gamma * self.lambda_
