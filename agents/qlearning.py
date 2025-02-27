import sys
import numpy as np


class Qlearning():

    def __init__(self, env, state_space, action_space,
                 alpha=0.3, gamma=1, eps=0.1):
        self.env = env
        self.alpha = alpha
        self.gamma = gamma
        self.eps = eps
        self.Q = np.ones((state_space, action_space)) * -100

    def select_action(self, state, greedy=False):
        if np.random.uniform(0, 1) < self.eps and not greedy:
            a = self.env.action_space.sample()
        else:
            a = np.argmax(self.Q[state, :])
        return a

    def update(self, transition):
        s, action, r, nxt_s, done = transition
        q_sa = self.Q[s, action]
        max_nxt_q = np.max(self.Q[nxt_s, :])
        # make the update
        new_q_sa = q_sa + self.alpha * (r + (1 - int(done)) * self.gamma * max_nxt_q - q_sa)
        self.Q[s, action] = new_q_sa

