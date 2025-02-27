from rooms_for_hrl import Fourrooms
from agents.high_levels import QlearningHL
from agents.low_level import QlearningLL
import numpy as np
import sys
sys.path.append("../")
from misc import print_info


def run(maze, algo, stochasticity, max_eps):

    # goal space can either be reduced or extended
    env = Fourrooms(maze, ep_length=1000, sub_length=10,
                    action_prob=stochasticity)
    lagent = QlearningLL(env, state_space=(4, maze['rooms'], 25), eps=0.0)
    hagent = QlearningHL(env, num_rooms=maze['rooms'], eps=0.4, mode=algo)
    ep_steps = np.zeros((max_eps, ))
    for ep in range(max_eps):
        sgs = []
        steps = 0
        obs = env.reset()
        high_done = False
        # decaying exploration and learning rate
        hagent.eps = hagent.eps * (1-1e-5)**ep
        lagent.eps = lagent.eps * (1-1e-7)**ep
        lagent.alpha = lagent.alpha * (1-1e-8)**ep
        while not high_done:
            low_done = False
            low_reward = 0
            room = obs[0]
            sg = hagent.select_subgoal(room)
            sgs.append(sg)
            env.set_goal(subgoal=sg)
            while not low_done:
                steps += 1
                low_level_obs = (sg,) + obs
                # TODO: weird doesn't work in this configuration.
                a = lagent.select_action(low_level_obs)
                nxt_obs, _, _, info = env.step(a)
                low_r = info['low reward']
                low_done = info['low done']
                high_r = info['high reward']
                high_done = info['high done']
                # update of Q value
                transition = (low_level_obs, a, low_r, (sg,) + nxt_obs, low_done)
                lagent.update(transition)
                obs = nxt_obs
                low_reward += low_r
                if high_done:
                    break
            high_transition = (room, sg, high_r, obs[0], high_done, low_done)
            hagent.update(high_transition)
        ep_steps[ep] = steps
        print(f"{ep}: {steps}")
    return ep_steps

if __name__ == '__main__':
    algos = ['qlearning']
    from mazes.config import *
    maze_names = [maze1x2, maze2x2, maze4x4]
    stochasticities = [1/3, 1/5]
    seeds = np.arange(10)
    max_eps = 3000
    save = True

    for algo in algos:
        for maze in maze_names:
            expname = f"hrl_large_low_level_state_space_{algo}_{maze['rooms']}_rooms"
            print(expname)
            steps = np.zeros((len(seeds), len(stochasticities), max_eps))
            for si, seed in enumerate(seeds):
                np.random.seed(seed)
                for pi, p in enumerate(stochasticities):
                    steps[si, pi, :] = run(maze, algo, p, max_eps)
                    if save:
                        np.save(f"./results/{expname}", steps)
