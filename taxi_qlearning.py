import gym
from agents.qlearning import Qlearning
import numpy as np


def print_ep(ep, max_ep, window, results, epsilon):
    avg_reward = np.mean(results[ep-window:ep+1, 0])
    avg_step = np.mean(results[ep-window:ep+1, 1])
    if ep % (10 * window) == 0 or ep == window:
        print()
        print(f"{'Episode' :<12}{'Reward' :<12}{'Step' :<12}{'Exploration' :<12}")
    episode = f"{ep}/{max_ep}".ljust(12)
    rew = f"{avg_reward:.2f}".ljust(12)
    step = f"{avg_step:.2f}".ljust(12)
    print(f"{episode}{rew}{step}{epsilon:.4f}")


def run(max_eps, seed, decay_epsilon,
        epsilon, print_every=10):
    env = gym.make('Taxi-v3')
    env.seed(seed)
    env.action_space.seed(seed)
    np.random.seed(seed)
    agent = Qlearning(env=env, state_space=500, action_space=6, eps=epsilon)
    results = np.zeros((max_eps, 3))  # rewards, steps, successes
    curr_ep = 0
    for ep in range(max_eps):
        obs = env.reset()
        reward = 0
        # decaying exploration and learning rates
        agent.eps = agent.eps * decay_epsilon**ep
        done = False
        step = 0
        while not done:
            a = agent.select_action([obs])
            nxt_obs, r, done, _ = env.step(a)
            # Q-update
            transition = (obs, a, r, nxt_obs, done)
            agent.update(transition)
            obs = nxt_obs
            reward += r
            step += 1
        results[curr_ep, 0] = reward
        if curr_ep % print_every == 0 and curr_ep > 0:
            print_ep(ep, max_eps, print_every, results, agent.eps)
        curr_ep += 1
        if r == 20:  # last reward is a success
            results[ep, 2] = 1
        results[ep, 1] = step
    return results


if __name__ == "__main__":
    N = 1
    env = gym.make('Taxi-v3')
    state = env.reset()
    taxi_row, taxi_col, passenger, dest = list(env.decode(state))
    print(f"Passenger {passenger} and destination {dest}")
    env.render()
    results = run(N, 1, 1-1e-3, 1-1e-7, epsilon=1)
    import matplotlib.pyplot as plt
    plt.plot(np.arange(N), results[:, 1])
    plt.show()
