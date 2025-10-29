import numpy as np
from mdptools.mdp import MDP
from mdptools.algorithms.value_iteration import value_iteration

"""
A simple example to demonstrate value iteration on a small MDP.

A robot can be in one of two states:
- State 0: At Home
- State 1: At Work

The robot can take one of two actions:
- Action 0: Stay
- Action 1: Move

The transition probabilities and rewards are defined as follows:
- From State 0:
  - Action 0 (Stay): 100% chance to stay in State 0, reward 1
  - Action 1 (Move): 100% chance to go to State 1, reward 5
- From State 1:
  - Action 0 (Stay): 100% chance to stay in State 1, reward 3
  - Action 1 (Move): 100% chance to go to State 0, reward 2

The discount factor is set to 0.9.
"""

S = np.array([[0], [1]])
A = np.array([[0], [1]])
P = np.array(
    [
        [
            [1.0, 0.0],
            [0.0, 1.0],
        ],
        [
            [0.0, 1.0],
            [1.0, 0.0],
        ],
    ]
)
R = np.array(
    [
        [1.0, 5.0],
        [3.0, 2.0],
    ]
)
gamma = 0.9

mdp = MDP(S, A, P, R, gamma)
V, policy = value_iteration(mdp)
print("Optimal Policy:\n", policy)
print("Optimal Value Function:\n", V)
