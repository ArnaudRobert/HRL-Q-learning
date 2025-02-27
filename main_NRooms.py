"""
Experiment comparing HSQL to Q-learning on the custom N-rooms environment.

A. Robert
"""
from exp_hrl import run as run_hrl
from baseline import run as run_baseline
from configs import configs
import numpy as np
import os
import time


num_profiles = [1]  # [1, 2, 3, 4]
maze_sizes = [4, 8, 16]    # [2, 4, 8, 16, 32]
num_seeds = 10
stochasticity = 1/5
algo = "stationary"
max_eps = 3000 # 3000 by default 5000 for 32x32 rooms
run_exp_hrl = True
run_exp_baseline = True
save = False


def save_results(steps, expname, method):
    path = "sanity_checks"
    filename = os.path.join(path, method+"_"+expname)
    np.save(filename, steps)


start = time.time()
for maze in maze_sizes:
    for room in num_profiles:
        name = f"{maze}{maze}{room}"
        config = configs[name]
        if run_exp_hrl:
            steps_hrl = np.zeros((num_seeds, max_eps))
        if run_exp_baseline:
            steps_baseline = np.zeros((num_seeds, max_eps))
        sub_length = config["sub length"]
        ep_length = config["episode length"]
        for seed in np.arange(num_seeds):
            # run Q-learning
            if run_exp_baseline:
                steps_baseline[seed, :] = run_baseline(config, stochasticity, max_eps, seed,
                                                       ep_length)
                if save:
                    save_results(steps_baseline, name, "baseline")
            # run the HRL algo
            if run_exp_hrl:
                steps_hrl[seed, :], _, _, _ = run_hrl(config, algo, stochasticity, max_eps, seed,
                                                  ep_length, render=False,
                                                      fix_reward=False)
                if save:
                    save_results(steps_hrl, name, "hrl")


end = time.time()
print(f"total running time is {end - start}")
