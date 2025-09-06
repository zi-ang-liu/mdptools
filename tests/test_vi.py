import numpy as np
from mdptools.mdp import MDP
from mdptools.algorithms.value_iteration import value_iteration

P = np.array([[[0.8, 0.2], [0.1, 0.9]], [[0.7, 0.3], [0.4, 0.6]]])
R = np.array([[5, 10], [0, 2]])
mdp = MDP(P, R, gamma=0.9)
V, policy = value_iteration(mdp)
print("Optimal Value Function:", V)
print("Optimal Policy:", policy)
