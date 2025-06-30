import numpy as np

class QLearningAgent:
    def __init__(self, maze_shape, n_actions=4, alpha=0.1, gamma=0.9, epsilon=0.1):
        self.q_table = {}  # {(y, x): [Q1, Q2, Q3, Q4]}
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.n_actions = n_actions
        self.maze_shape = maze_shape

    def get_qs(self, state):
        if state not in self.q_table:
            self.q_table[state] = np.zeros(self.n_actions)
        return self.q_table[state]

    def choose_action(self, state):
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.n_actions)
        return np.argmax(self.get_qs(state))

    def update(self, state, action, reward, next_state):
        current_q = self.get_qs(state)[action]
        max_future_q = np.max(self.get_qs(next_state))
        new_q = current_q + self.alpha * (reward + self.gamma * max_future_q - current_q)
        self.q_table[state][action] = new_q
