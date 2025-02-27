"""
Function that runs Q-learning in the taxi environment.

A. Robert
"""
from gym import Env
from envs.taxi_hrl import HierarchicalTaxi
from agents.low_level import QlearningLL
from agents.high_levels import QlearningHL
from agents.base import HL, LL
import numpy as np


def print_ep(ep, max_ep, print_every, results):
    lowR = results[ep, 0]
    highR = results[ep, 1]
    lowEps = results[ep, 2]
    highEps = results[ep, 3]
    steps = results[ep, 7]

    if ep % (10 * print_every) == 0 or ep == print_every:
        print()
        print(f"{'Episode' :<12}{'Low reward' :<12}{'High reward' :<15}{'Low exploration': <17}{'High exploration': <20}{'Subgoals': <10}{'Step' :<6}")
    episode_ = f"{ep}/{max_ep}".ljust(12)
    lowR_ = f"{lowR:.2f}".ljust(12)
    highR_ = f"{highR:.2f}".ljust(15)
    lowEps_ = f"{lowEps:.3f}".ljust(17)
    highEps_ = f"{highEps:.3f}".ljust(20)
    steps_ = f"{steps}".ljust(6)
    subgoals_ = f"{results[ep, 5]}/{results[ep, 4]}".ljust(10)
    print(f"{episode_}{lowR_}{highR_}{lowEps_}{highEps_}{subgoals_}{steps_}")


def hardcoded_HL(p_loc, e_goal):
    """
    For debugging purposes, an optimal implementation of the high-level agent
    in the Hierarchical Taxi Environment.
    """
    if p_loc != 4:                          # if the passenger is not "in Taxi"
        aH = p_loc
    else:                                   # if passenger is "in Taxi"
        aH = 4 + e_goal
    return aH


def random_HL(p_loc, e_goal):
    """
    For debugging purposes, an optimal implementation of the high-level agent
    in the Hierarchical Taxi Environment.
    """
    if p_loc != 4:                          # if the passenger is not "in Taxi"
        aH = p_loc
    else:                                   # if passenger is "in Taxi"
        aH = 4 + np.random.randint(low=0, high=4)
    return aH


def run_episode(env: Env, low_agent: LL, high_agent: HL,
                curr_ep: int, decayLEps: float, decayHEps: float):
    """
    Run a single episode of hierarchical learning
    """
    # lowR, highR, lowEps, highEps, num_subgoal, subgoal solved,success and steps
    monitoring = np.zeros((8,))
    highDone = False
    lowS, highS, info = env.reset()
    while not highDone:
        # set subgoal
        aH = high_agent.select_subgoal(highS)
        # aH = hardcoded_HL(info['Passenger loc'], info["End goal"])
        # aH = random_HL(info['Passenger loc'], info["End goal"])
        lowS, highDone = env.set_subgoal(aH)
        if highDone:
            break
        lowDone = False
        monitoring[4] += 1  # count # of subgoals
        while not lowDone and not highDone:
            # select action
            aL = low_agent.select_action(lowS)
            nxtLowS, nxtHighS, lowR, highR, lowDone, highDone, info = env.step(aL)
            transition = (lowS, aL, lowR, nxtLowS, lowDone)

            # update low-level
            low_agent.update(transition)
            lowS = nxtLowS

            # monitor experiment
            monitoring[0] += lowR
            if info['Passenger loc'] == info['End goal']:
                monitoring[6] += 1  # count # of solved tasks

            if lowR == 20:
                monitoring[5] += 1  # count # of solved subgoals

            monitoring[7] += 1  # count # of steps

        # update high level agent
        high_transition = (highS, aH, highR, nxtHighS, highDone, lowDone)
        high_agent.update(high_transition)
        highS = nxtHighS

        # monitor experiment
        monitoring[1] += highR
        low_agent.eps = low_agent.eps * decayLEps**curr_ep
        monitoring[2] = low_agent.eps

    # end of the episode
    high_agent.eps = high_agent.eps * decayHEps**curr_ep
    monitoring[3] = high_agent.eps
    return monitoring


def run(num_episodes, seed, decayLEps, decayHEps, print_every=10):
    env = HierarchicalTaxi()
    env.seed(seed)
    low_agent = QlearningLL(env.low_action_space,
                            env.low_state_space.n,
                            env.low_action_space.n,
                            env.high_action_space.n,
                            eps=0.8)
    high_agent = QlearningHL(env, env.high_state_space.n,
                             env.high_action_space.n, eps=0.8)
    results = np.zeros((num_episodes, 8))
    for ep in range(num_episodes):
        results[ep, :] = run_episode(env, low_agent, high_agent, ep, decayLEps,
                                     decayHEps)
        if ep % print_every == 0 and ep > 0:
            print_ep(ep, num_episodes, print_every, results)
    return results

if __name__ == "__main__":
    run(100, 123, 1-1e-6, 1-1e-7)
