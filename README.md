# THRL

Code used to reproduce the experiments in the article "Sample Complexity of Goal-Conditioned Hierarcical Reinforcement Learning". 

The script 'main.py' runs the set of experiments presented in the article.

The code is organised as follow:
- baseline.py contains the script to run Q-learning in the environments.
- exp_hrl.py contains the script to run SHQL in the environments.
- nRoomsEnv.py contains the code to interact witn the environment (to create the mazes fct. in generate_mazes.py)
- the code for Q-learning and SHQL agents is in the folder agent. 
