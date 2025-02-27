"""
Experiment comparing HSQL to Q-learning
on the Taxi environment

A. Robert
"""
import time
import numpy as np
from taxi_qlearning import run as run_qlearning
from taxi_hsql import run as run_hsql
###########################
# Experiment parameters
###########################
num_seeds = 10
hsql = False
qlearning = True
num_episodes = 5000
save = True

# Params for Q-learning
#decay_alpha = 1.
decay_epsilon = 1 - 1e-5
epsilon = 0.8

# Params for HSQL
decay_low_eps = 1 - 1e-5
decay_high_eps = 1 - 1e-5


def print_experiment_details(num_seeds, seed, method):
    print()
    print("*********************************")
    print(f"* Running {method} seed {seed}/{num_seeds}")
    print("*********************************")
    print()
    time.sleep(1.5)


def save_results(steps, rewards, expname, algo):
    pass


def run(epsilon, decay, decay_low, decay_high, hsql, qlearning, save):
    name = f"./results/taxi/taxi_results_eps_{epsilon}_decay_{decay}"
    if hsql:
        hsql_results = np.zeros((num_episodes, 8, num_seeds))
    if qlearning:
        # results are episode rewards, steps and successes
        qlearning_results = np.zeros((num_episodes, 3, num_seeds))

    for seed in range(num_seeds):
        if hsql:
            print_experiment_details(num_seeds, seed, "HSQL")
            hsql_results[:, :, seed] = run_hsql(num_episodes,
                                                seed,
                                                decay_low,
                                                decay_high)
            if save:
                np.save(name+'_hsql', hsql_results)
        if qlearning:
            print_experiment_details(num_seeds, seed, "Q-learning")
            qlearning_results[:, :, seed] = run_qlearning(num_episodes,
                                                          seed,
                                                          decay,
                                                          epsilon)
            if save:
                np.save(name+'_qlearning', qlearning_results)


def grid_search(hsql, qlearning, save):
    epsilons = [0.1, 0.3, 0.5, 0.7, 0.9]
    decays = [1-1e-3, 1-1e-4, 1-1e-5, 1-1e-6, 1-1e-7]

    for eps in epsilons:
        for decay in decays:
            run(eps, decay, decay, decay, hsql, qlearning, save)


if __name__ == "__main__":
    grid_search(True, True, True)

