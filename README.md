# THRL

Code used to reproduce the experiments in the article "Sample Complexity of Goal-Conditioned Hierarchical Reinforcement Learning", published at NeurIPS 2024.

The script 'main.py' runs the set of experiments presented in the article.

The code is organised as follows:
- baseline.py contains the script to run Q-learning in the environments.
- exp_hrl.py contains the script to run SHQL in the environments.
- nRoomsEnv.py contains the code to interact with the environment
- generate_mazes.py contains the necessary function to create mazes.
- The code for Q-learning and SHQL agents is in the agent folder. 
