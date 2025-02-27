from agents.base import LL
import numpy as np
from gym.spaces import Space


class QlearningLL(LL):

    def __init__(self, action_space: Space, nState: int, nAction: int,
                 nGoal: int, alpha: float = 0.3, gamma: float = 1.,
                 eps: float = 0.1, use_subgoals: bool = False):
        """
        """
        self.action_space = action_space
        self.nAction = nAction
        self.nState = nState
        self.nGoal = nGoal
        self.alpha = alpha
        self.gamma = gamma
        self.eps = eps
        if nGoal > 0:
            self.Q = np.ones((self.nState, self.nGoal) + (self.nAction,)) * 1e-6
        else:
            self.Q = np.ones(self.nState + (self.nAction,)) * 1e-6

    def select_action(self, state, greedy=False):
        z = None
        x = state[0]
        y = state[1]
        if len(state) > 2:
            z = state[2]
        if np.random.uniform(0, 1) < self.eps and not greedy:
            a = self.action_space.sample()
        else:
            if z is None:
                a = np.argmax(self.Q[x, y, :])
            else:
                a = np.argmax(self.Q[x, y, z, :])
        return a

    def get_V(self, g):
        V = np.zeros((len(self.state_space),))
        for s in np.arange(len(self.state_space)):
            V[s] = self.Q[s, g, :].max()
        return V

    def get_score(self, s, goal):
        S = np.zeros((len(self.goal_space),))
        for g in np.arange(len(self.goal_space)):
            S[g] = self.Q[s, g, :].max()
        return np.exp(S[goal]) / np.sum(np.exp(S))

    def print_policy(self, goal, room=None):
        if room is None:
            action = np.argmax(self.Q[goal, :, :], axis=1)
        else:
            action = np.argmax(self.Q[goal, room, :, :], axis=1)
        print(action.reshape(5, 5))

    def get_goal_preferences(self, s, psi=10):
        proba = []
        for g in np.arange(len(self.goal_space)):
            proba.append(psi*self.Q[s, g, :].max())
        return np.exp(proba) / np.sum(np.exp(proba))

    def update(self, transition):
        s, action, r, nxt_s, done = transition
        z = nxt_z = None
        if len(s) > 2:
            z = s[2]
            nxt_z = nxt_s[2]
        x = s[0]
        y = s[1]
        nxt_x = nxt_s[0]
        nxt_y = nxt_s[1]
        if z is None:
            q_sa = self.Q[x, y, action]
            max_nxt_q = np.max(self.Q[nxt_x, nxt_y, :])
        else:
            q_sa = self.Q[x, y, z, action]
            max_nxt_q = np.max(self.Q[nxt_x, nxt_y, nxt_z, :])
        # make the update
        new_q_sa = q_sa + self.alpha * (r + (1 - int(done)) * self.gamma * max_nxt_q - q_sa)
        if z is None:
            self.Q[x, y, action] = new_q_sa
        else:
            self.Q[x, y, z, action] = new_q_sa


