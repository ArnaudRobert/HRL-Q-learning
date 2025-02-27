import logging
import math
import gym
from gym import spaces
from gym.utils import seeding
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Arrow, Circle
logger = logging.getLogger(__name__)


class NRooms(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 50
    }

    def __init__(self, config, seed=1234,
                 rg=False, ep_length=1000,
                 sub_length=10, action_prob=1/3,
                 fix_reward=True):
        """
        Load the maze and create the associated environment
        input:
            -config      : the config file describing the maze
            -seed        : the random seed governing stochasticity of env.
            -ep_length   : the length of an episode
            -sub_length  : the length of a sub-episode
            -action_prob : the probability of an action
            -fix_reward  : the type of high-level reward function
                           if true the high-level reward is a fix
                           constant. otherwise it is the same as in
                           the article.
        """

        self.fix_reward = fix_reward
        self.first = True

        # read the config file for the maze
        self.room_width = config['maze width']
        self.room_height = config['maze height']
        self.goal = config['goal loc']
        self.maze = np.load(config['maze'])
        self.rooms = np.load(config['rooms'])
        self.rooms -= 1
        self.rooms[self.rooms < 0] = -1
        self.states = np.load(config['states'])
        self.width, self.height = self.maze.shape

        # set env constant
        self.action_prob = action_prob
        self.max_steps = ep_length
        self.max_sub_steps = sub_length
        self.rdn_goal = rg

        # compute maze representation
        tmp = np.unique(self.rooms)
        self.num_rooms = len(tmp[tmp>=0])
        tmp = np.unique(self.states)
        self.num_states = len(tmp[tmp>=0])

        # states are (room number, location in the room)
        self.tostate = {} # (R, LOC) -> (i,j)
        for i in range(self.maze.shape[0]):
            for j in range(self.maze.shape[1]):
                if self.maze[i, j] == 1:  # this a cell
                    state = self.states[i, j]
                    room = self.rooms[i, j]
                    self.tostate[(i, j)] = (int(room), int(state))
        self.tocell = {v: k for k,v in self.tostate.items()}
        self.action_space = spaces.Discrete(4)

        # actions meaning: 0 = N, 1 = S, 2 = W, 3 = E
        self.directions = [np.array((-1, 0)), np.array((1, 0)),
                           np.array((0, -1)), np.array((0, 1))]
        self.rng = np.random.RandomState(seed)
        self.action_space.np_random.seed(seed)

    def seed(self, seed=None):
        return self._seed(seed)

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def reset(self, cell=(1, 3)):
        self.currentcell = cell
        state = self.tostate[cell]
        self.last_door = "N"
        self.current_step = 0
        self.current_sub_step = 0
        return (state[0], state[1], self.last_door)

    def get_available_action(self, current_room):
        # get all the indices of a given room use self.rooms
        indices = np.transpose((self.rooms == current_room).nonzero())
        maxi, maxj = np.max(indices, axis=0)
        mini, minj = np.min(indices, axis=0)

        actions = set((0, 1, 2, 3))  # N, S, W, E
        if mini <= 1:
            actions.remove(0)  # remove N
        if minj <= 1:
            actions.remove(2)  # remove W
        if maxj >= self.width - 2:
            actions.remove(3)  # remove E
        if maxi >= self.height - 2:
            actions.remove(1)  # remove S
        #print(f"In room {current_room} avaible actions are {actions}")
        return list(actions)

    def goal2room(self, current_state, goal):
        # use self.rooms for that
        room = current_state[0]
        indices = np.transpose((self.rooms == room).nonzero())
        maxi, maxj = np.max(indices, axis=0)
        mini, minj = np.min(indices, axis=0)
        midi = mini + (maxi - mini)//2
        midj = minj + (maxj - minj)//2
        if goal == 0:  # exit room from north
            target = int(self.rooms[mini - 2, midj])

        elif goal == 3:  # exit room from east
            target = int(self.rooms[midi, maxj+2])

        elif goal == 1:  # exit room from south
            target = int(self.rooms[maxi + 2, midj])

        else:  # exit room from west
            target = int(self.rooms[midi, minj-2])
        #print(f"current room {room} - goal is {goal} - target room {target}")
        #input()
        return target

    def plot_Q_hl(self, Q, ax):
        viz = self.maze * -1
        viz[viz == 1] = 0
        ax.imshow(viz, cmap='Greys')

    def render(self, state, sg, Q_hl=None, show_goal=True):
        current_grid = np.array(self.maze)
        current_grid[self.currentcell[0], self.currentcell[1]] = -1
        if show_goal:
            goal_cell = self.tocell[(self.goal[0], self.goal[1])]
            current_grid[goal_cell[0], goal_cell[1]] = -2

        if self.first:
            self.first = False
            from matplotlib import pyplot as plt
            plt.ion()
            self.fig, self.ax = plt.subplots(1, 2)
            self.axim = self.ax[0].imshow(current_grid)
        self.plot_Q_hl(Q_hl, self.ax[1])
        goal2text = {0: 'N', 1: 'S', 2: 'W', 3: 'E'}
        self.ax[0].set_title(f"Currently in room {state[0]} cell {state[1]} coming from exit {state[2]}, current sub-goal: {goal2text[sg]}")
        print(f"Currently in room {state[0]} cell {state[1]} coming from exit {state[2]}, current sub-goal: {goal2text[sg]}")
        self.axim.set_data(current_grid)
        self.axim.set_data(current_grid)
        self.fig.canvas.flush_events()
        return current_grid

    def _step(self, cell, direction):
        new_room = False
        nextcell = cell + direction
        i = nextcell[0]
        j = nextcell[1]
        if self.maze[i, j] == 0: # hit a wall nothing happen
            nextcell = cell
        elif self.maze[i, j] == -1: # pass a door
            nextcell += direction
            new_room = True
        else:
            pass
        return nextcell, new_room

    def set_goal(self, subgoal=None, goal=None):
        if not goal is None:
            self.goal_room = goal
        elif not subgoal is None:
            state = self.tostate[(self.currentcell[0], self.currentcell[1])]
            self.goal_room = self.goal2room(state, subgoal)

    def step(self, action):
        """
        The agent can perform one of four actions,
        up, down, left or right, which have a stochastic effect.
        With probability 2/3, the actions
        cause the agent to move one cell in the corresponding direction,
        and with probability 1/3,
        the agent moves instead in one of the other three directions,
        each with 1/9 probability. In either case,
        if the movement would take the agent into a wall then the agent
        remains in the same cell.
        We consider a case in which rewards are zero on all state transitions.

        Recall the actions meaning: 0 = N, 1 = S, 2 = W, 3 = E
        """
        self.current_step += 1
        self.current_sub_step += 1
        high_done = False
        low_done = False

        if np.random.uniform() < self.action_prob:
            action = np.random.randint(low=0, high=4)

        self.currentcell, new_room = self._step(self.currentcell, self.directions[action])
        state = self.tostate[(self.currentcell[0], self.currentcell[1])]

        if new_room: # change entry point
            if action == 0:
                self.last_door = "S"
            if action == 1:
                self.last_door = "N"
            if action == 2:
                self.last_door = "E"
            if action == 3:
                self.last_door = "W"

        low_reward = -0.1

        if self.fix_reward:
            high_reward = -0.1
        else:
            high_reward = 2*low_reward * (self.current_sub_step+1)

        if state[0] == self.goal[0]:
            high_reward = 0
            high_done = True

        if state[0] == self.goal_room:
            low_reward = 0
            low_done = True
            self.current_sub_step = 0

        if self.current_step > self.max_steps:
            high_done = True

        if self.current_sub_step > self.max_sub_steps:
            low_done = True
            self.current_sub_step = 0

        info = {'low reward': low_reward,
                'low done': low_done,
                'high reward': high_reward,
                'high done': high_done}

        return (state[0], state[1], self.last_door), None, None, info

    def show_value_function(self, s, V, ax):
        ax.imshow(self.occupancy, cmap='Greys')
        values = np.ones_like(self.occupancy) * np.NAN
        for g, v in enumerate(V):
            if g == s:
                (i, j) = self.tocell[s]
                ax.scatter(j, i, color="red")
            else:
                (i, j) = self.tocell[g]
                values[i, j] = v #np.exp(v)/np.sum(np.exp(V))
        im = ax.imshow(values, cmap=plt.cm.viridis, alpha=.9, vmin=0, vmax=1)
        return ax

    def show_env(self):
        # x, y = self.tocell[self.goal]
        fig, ax = plt.subplots(1, 1)
        viz = self.maze.copy()
        viz[self.maze == 1] = 0
        viz[self.maze == -1] = 0
        viz[self.maze == 0] = 1
        ax.imshow(viz, cmap="Greys")
        # goal = Circle((y, x), radius=0.2, color='red')
        start = Circle((1, 1), radius=0.2, color='blue')
        # ax.add_patch(goal)
        ax.add_patch(start)
        plt.show()


if __name__ == "__main__":
    from configs import *
    env = NRooms(maze2x2_7x7_3)
    # env.show_env()
    # env.show_contexts()
    s = env.reset()
    print(s)
