"""
Notes on the environement.

We consider the following hierarchical decomposition of the taxi environment.

High-level MDP:
    Action space:    Of size 8 encodes the pickup and dropoff locations
                     0: pickup@R, 1: pickup@G, 2: pickup@Y, 3: pickup@B,
                     4: dropoff@R, 5: dropoff@G, 6: dropoff@Y, 7: dropoff@B

    State space:     Of size 20:
                     20 = 5 (current passenger loc.) x 4 (final destination)

    Episode length:  Of length 2

Low-level MDP:
    Action space:    The 6 primitive actions

    State space:     Of size 400 = 8 (sub-goals) x 25 (taxi loc.).

    Episode length:  Of length 100

A. Robert
"""
import gym
from gym import Env
from gym.spaces import Discrete
import numpy as np

class HierarchicalTaxi(Env):
    """
    Wrapper of the taxi environment that decompose
    the original MDP into the corresponding low-level and
    high-level MDPs
    """

    def __init__(self):
        self._env = gym.make('Taxi-v3')
        self.low_action_space = self._env.action_space
        self.low_state_space = Discrete(25)
        self.high_action_space = Discrete(8)
        self.high_state_space = Discrete(20)
        self.t = 0
        self.high_t = 0
        self.high_H = 2
        self.low_H = 100
        self.low_state = [None, None]
        self.high_state = None
        self.high_a = None
        self.acc_r = 0

    def seed(self, seed=None):
        self.low_state_space.seed(seed)
        self.high_action_space.seed(seed)
        self.low_action_space.seed(seed)
        self._env.seed(seed)
        np.random.seed(seed)

    def low_level_reward(self, passenger, r):
        """
        The low-level is rewarded if the passenger
        reach the location instructed by sg.

        Input:
            high_s : the high-level state
            sg     : the current subgoal
            r      : the environment reward
        """
        done = False
        if self.high_a < 4 and passenger == 4:  # pickup instruction 
            reward = 20
            done = True
        elif self.high_a >= 4 and (self.high_a % 4) == passenger:
            reward = 20
            done = True
        elif r < 0:
            reward = r
        else:
            reward = -1
        return reward, done

    def step(self, low_action):
        """
        The low-level's action is used to interact with
        the environment and the high-level's action
        or sub-goal information are used to compute rewards.

        Inputs:
            low_action :  the action used to interact with
                          the environment
            high_action : the high-level action is the current
                          sub-goal.
        Outputs:
            low_state   : the low-level state description
            high_state  : the high-level state description
            low_r       : the low-level reward
            high_r      : the high-level reward
            low_done    : flag the end of a low-level episode
            high_done   : flag the end of a high-level episode
        """
        state, r, high_done, info = self._env.step(low_action)
        low, high, info = self._build_obs(state)
        self.low_state = [low, self.high_a]
        self.high_state = high

        # compute rewards
        low_r, low_done = self.low_level_reward(info["Passenger loc"], r)
        if self.t % self.low_H == 0 and self.t > 0:
            low_done = True

        self.t += 1
        self.acc_r += low_r

        if info["Passenger loc"] == info["End goal"]:
            high_r = 100
            high_done = True
        else:
            high_r = -1

        return self.low_state, self.high_state, low_r, self.acc_r, low_done, high_done, info

    def _build_obs(self, state):
        taxi_row, taxi_col, passenger, dest = list(self._env.decode(state))
        # the sub_goal is added later in low-level state
        low = 5 * taxi_col + taxi_row
        # build high level state
        high = dest * 5 + passenger
        info = {"End goal": dest,
                "Passenger loc": passenger,
                "Current sub-goal": self.high_a,
                "Taxi row": taxi_row,
                "Taxi col": taxi_col}
        return low, high, info

    def set_subgoal(self, sg):
        """
        The sub-goal indicate the next desired passenger location.
        The various subgoals are encoded as follow:

        0: pickup@R   4: dropoff@R
        1: pickup@G   5: dropoff@G
        2: pickup@Y   6: dropoff@Y
        3: pickup@B   7: dropoff@B

        Input:
            - sg : The selected sub-goal

        Output :
            - low_state: The updated low-level state
        """
        self.high_a = sg
        self.low_state[1] = sg
        self.acc_r = 0
        self.high_t += 1
        if self.high_t > 2:
            highDone = True
        else:
            highDone = False
        return self.low_state, highDone

    def render(self):
        self._env.render()

    def get_available_action(self, highS):
        #TODO: extract passenger location from highS
        passenger = highS % 5
        if passenger < 4:
            actions = set((passenger, ))
        else:
            actions = set((4, 5, 6, 7))
        return list(actions)

    def reset(self):
        state = self._env.reset()
        self.t = 0
        self.high_t = 0
        low, high, info = self._build_obs(state)
        self.high_state = high
        self.low_state[0] = low
        return self.low_state, self.high_state, info


if __name__ == '__main__':
    env = HierarchicalTaxi()
    low_state, high_state, _ = env.reset()

    high_done = False
    while not high_done:
        low_done = False
        high_a = env.high_action_space.sample()
        low_state = env.set_subgoal(high_a)
        while not low_done:
            low_a = env.low_action_space.sample()
            low_state, high_state, low_r, high_r, low_done, high_done, info = env.step(low_a)
            env.render()
            if high_done:
                break
    print("End of the test episode")
