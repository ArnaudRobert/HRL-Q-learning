from exp_hrl import run as run_hrl
from baseline import run as run_baseline
from configs import configs
import numpy as np
import os

num_profiles = [1, 2, 3, 4]  #[1, 2, 3, 4]
maze_sizes =  [2, 4]  # [2, 4, 8, 16, 32]
num_seeds = 10
stochasticity = 1/5
algo = "stationary"
max_eps = 3000 # 5000 for 32x32 rooms 
save = True


def save_results(steps, expname, method):
    path = "results/compare_reward_functions"
    filename = os.path.join(path, method+"_"+expname)
    np.save(filename, steps)


for maze in maze_sizes:
    for room in num_profiles:
        name = f"{maze}{maze}{room}"
        config = configs[name]
        steps_hrl = np.zeros((num_seeds, max_eps))
        sub_length = config["sub length"]
        ep_length = config["episode length"]
        for seed in np.arange(num_seeds):
            for fix_reward in [False]:
                steps_hrl[seed, :], _, _, _ = run_hrl(config, algo,
                                                      stochasticity,
                                                      max_eps, seed,
                                                      ep_length,
                                                      render=False,
                                                      fix_reward=fix_reward)
                if save:
                    if fix_reward:
                        tmp = "fix_reward"
                    else:
                        tmp = "cumulated_reward"
                    save_results(steps_hrl, name, "hrl_"+tmp)

