from envs.nRoomsEnv import NRooms
from agents.high_levels import QlearningHL
from agents.low_level import QlearningLL
import numpy as np
import sys
sys.path.append("../")


def run(maze, stochasticity, max_eps, seed, ep_length):

    # goal space can either be reduced or extended
    env = NRooms(maze, ep_length=ep_length, sub_length=ep_length,
                 action_prob=stochasticity)
    np.random.seed(seed)
    num_rooms = env.num_rooms
    num_states = env.num_states
    lagent = QlearningLL(action_space=env.action_space,
                         nState=(num_rooms, num_states),
                         nAction=4, nGoal=0, eps=0.3)
    ep_steps = np.zeros((max_eps, ))
    for ep in range(max_eps):
        steps = 0
        obs = env.reset()
        obs = obs[:-1]
        high_done = False
        # decaying exploration and learning rate
        lagent.eps = lagent.eps * (1-1e-5)**ep
        lagent.alpha = lagent.alpha * (1-1e-8)**ep
        low_done = False
        low_reward = 0
        env.set_goal(goal=env.goal[0])
        while not low_done:
            steps += 1
            a = lagent.select_action(obs)
            nxt_obs, _, _, info = env.step(a)
            nxt_obs = nxt_obs[:-1]
            low_r = info['low reward']
            low_done = info['low done']
            high_r = info['high reward']
            high_done = info['high done']
            # update of Q value
            transition = (obs, a, low_r, nxt_obs, low_done)
            lagent.update(transition)
            obs = nxt_obs
            low_reward += high_r

            if high_done:
                break
        ep_steps[ep] = steps
        print(f"{ep}: {steps}")

    return ep_steps


if __name__ == '__main__':
    from configs import *
    maze_names = [maze2x2_7x7_1, maze4x4_7x7_1, maze8x8_7x7_1,
                  maze16x16_7x7_1, maze32x32_7x7_1]
    maze_names = [maze8x8_7x7_2]
    stochasticities = [1/5]
    seeds = np.arange(10)
    max_eps = 3000
    save = True

    for maze in maze_names:
        name = maze['rooms'].split('/')[1]
        name = name.split('.')[0]
        expname = f"flat_{name}"
        steps = np.zeros((len(seeds), len(stochasticities), max_eps))
        for si, seed in enumerate(seeds):
            for pi, p in enumerate(stochasticities):
                if maze == maze32x32_7x7_1 or maze == maze32x32_7x7_2 or maze == maze32x32_7x7_3 or maze == maze32x32_7x7_4:

                    ep_length = 10000
                elif maze == maze16x16_7x7_1 or maze == maze16x16_7x7_2 or maze == maze16x16_7x7_3 or maze == maze16x16_7x7_4:
                    ep_length = 5000
                else:
                    ep_length = 1000
                steps[si, pi, :] = run(maze, p, max_eps, ep_length)
                if save:
                    np.save(f"./results/{expname}", steps)
