from envs.nRoomsEnv import NRooms
from agents.high_levels import QlearningHL
from agents.low_level import QlearningLL
import numpy as np
from configs import *


def run(maze, algo, stochasticity, max_eps, seed,
        ep_length=1000, render=False, fix_reward=False):

    # goal space can either be reduced or extended
    env = NRooms(maze, ep_length=ep_length, sub_length=20,
                 action_prob=stochasticity, seed=seed,
                 fix_reward=fix_reward)
    ll_states = env.num_states
    hl_states = env.num_rooms

    lagent = QlearningLL(action_space=env.action_space,
                         nState=ll_states,
                         nAction=4, nGoal=4, eps=0.3)
    hagent = QlearningHL(env, nStates=hl_states, nActions=4,
                         eps=0.4, mode=algo)

    ep_steps = np.zeros((max_eps, ))
    use_updates = np.zeros((max_eps,))

    for ep in range(max_eps):
        sgs = []
        sg_solved = 0
        steps = 0
        obs = env.reset()
        high_done = False

        # decaying exploration and learning rate
        hagent.eps = hagent.eps * (1-5e-6)**ep
        lagent.eps = lagent.eps * (1-1e-5)**ep
        lagent.alpha = lagent.alpha * (1-1e-8)**ep
        hagent.alpha = hagent.alpha * (1-1e-8)**ep
        furthest = 0

        while not high_done:
            low_done = False
            low_reward = 0
            hl_obs = (obs[0])
            sg = hagent.select_subgoal(hl_obs)
            sgs.append(sg)
            env.set_goal(subgoal=sg)

            while not low_done:
                steps += 1
                loc = obs[1]
                low_level_obs = (loc, sg)
                a = lagent.select_action(low_level_obs)
                nxt_obs, _, _, info = env.step(a)

                if render:
                    env.render(nxt_obs, sg)
                low_r = info['low reward']
                low_done = info['low done']
                sg_solved += int(low_done)
                high_r = info['high reward']
                high_done = info['high done']
                # update of Q value
                transition = (low_level_obs, a, low_r, (nxt_obs[1], sg), low_done)
                lagent.update(transition)
                obs = nxt_obs
                low_reward += low_r

                if high_done:
                    break

                if obs[0] > furthest:
                    furthest = obs[0]

            use_updates[ep] = int(low_done)
            high_transition = (hl_obs, sg, high_r, obs[0], high_done, low_done)
            hagent.update(high_transition)
        ep_steps[ep] = steps
        print(f"{ep}: steps: {steps} subgoals: {sg_solved}/{len(sgs)} - current epsilon {hagent.eps} - hl reward: {high_r}")
    return ep_steps, lagent, hagent, use_updates


if __name__ == '__main__':
    simple_maze_exp = True

    if simple_maze_exp:
        algos = ['stationary']
        # maze_names = [maze2x2_7x7_2, maze4x4_7x7_2, maze8x8_7x7_2,
        #             maze16x16_7x7_2, maze32x32_7x7_2]
        maze_names = [maze8x8_7x7_3]
    stochasticities = [1/5]
    seeds = np.arange(10)
    max_eps = 3000
    save = True

    for algo in algos:
        for maze in maze_names:
            name = maze['rooms'].split("/")[1]
            name = name.split('.')[0]
            expname = f"hrl_{algo}_{name}"
            steps = np.zeros((len(seeds), len(stochasticities), max_eps))
            for si, seed in enumerate(seeds):
                np.random.seed(seed)
                for pi, p in enumerate(stochasticities):
                    ep_length = maze['episode length']
                    ep_length = 2000
                    steps[si, pi, :], lagent, hagent, used_transitions = run(maze, algo, p,
                                                                             max_eps, seed,
                                                                             ep_length, False)
                    if save:
                        if False and algo=='strict':
                            np.save(f"./results/{expname}_used_transitions", used_transitions)
                        np.save(f"./results/{expname}", steps)
                        np.save(f"./results/{expname}_low_level_Q", lagent.Q)
                        np.save(f"./results/{expname}_high_level_Q", hagent.Q)

