import numpy as np
from mdptools.mdp import MDP
from mdptools.algorithms.value_iteration import value_iteration

S = np.array([[0], [1], [2]])
A = np.array([[0], [1]])
P = np.array(
    [
        [[0.8, 0.2, 0.0], [0.1, 0.9, 0.0]],
        [[0.0, 0.7, 0.3], [0.4, 0.6, 0.0]],
        [[0.0, 0.0, 1.0], [0.5, 0.5, 0.0]],
    ]
)
R = np.array([[5, 10], [2, 4], [0, 1]])
gamma = 0.9

mdp = MDP(S, A, P, R, gamma)
policy, V = value_iteration(mdp)

print("Optimal Policy:\n", policy)
print("Value Function:\n", V)
