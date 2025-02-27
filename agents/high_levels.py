import numpy as np
from agents.base import HL


class QlearningHL(HL):

    def __init__(self, env, nStates, nActions, alpha=0.3,
                 gamma=1, eps=0.5, mode='stationary', action_mask=True):
        self.env = env
        self.n_actions = nActions
        self.Q = np.ones((nStates, self.n_actions)) * 0

        self.alpha = alpha
        self.eps = eps
        self.gamma = gamma
        self.mode = mode
        self.action_mask = action_mask

    def select_subgoal(self, context, greedy=False):
        if self.action_mask:
            actions = self.env.get_available_action(context)
        else:
            actions = np.arange(self.n_actions)

        if np.random.uniform(0, 1) < self.eps and not greedy:
            sg = np.random.choice(actions)
        else:
            tmp = np.argmax(self.Q[context, actions])
            sg = actions[tmp]
        return sg

    def update(self, transition):
        if self.mode == 'qlearning':
            self._qlearning_update(transition)

        elif self.mode == 'stationary':
            _, _, _, _, _, low_done = transition
            if low_done:
                self._qlearning_update(transition)
        else:
            print(f"Unknow algorithm: {self.mode}")
            exit()

    def _qlearning_update(self, transition):
        x, sg, r, nxt_x, done, _ = transition
        q_sg = self.Q[x, sg]
        if self.action_mask:
            mask = self.env.get_available_action(nxt_x)
            max_nxt_q = np.max(self.Q[nxt_x, mask])
        else:
            max_nxt_q = np.max(self.Q[nxt_x, :])
        new_q_sg = q_sg + self.alpha * (r + (1 - int(done)) * self.gamma * max_nxt_q - q_sg)
        self.Q[x, sg] = new_q_sg

    def get_V(self):
        pass

    def print_policy(self):
        print(self.Q[:, :])

