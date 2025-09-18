import numpy as np
from mdptools.mdp import MDP
from mdptools.algorithms.value_iteration import value_iteration

"""
Sutton et al. (2018) Example 3.8

Gridworld:

actions: up, down, left, right

probabilities:
- intended direction: 100%

reward: 
- action to A: +10
- action to B: +5
- all other actions: 0
- moving off the grid: -1
- discount factor: 0.9
"""


def create_gridworld_mdp():
    n_rows, n_cols = 5, 5
    n_states = n_rows * n_cols
    n_actions = 4  # up, down, left, right
    state_A = (0, 1)
    state_B = (0, 3)
    state_A_prime = (4, 1)
    state_B_prime = (2, 3)
    S = np.array([(i, j) for i in range(n_rows) for j in range(n_cols)])
    A = np.array([0, 1, 2, 3])  # up, down, left, right

    def state_to_index(state):
        return state[0] * n_cols + state[1]

    def index_to_state(index):
        return (index // n_cols, index % n_cols)

    def is_valid(state):
        return 0 <= state[0] < n_rows and 0 <= state[1] < n_cols

    def move(state, action):
        if action == 0:  # up
            return (state[0] - 1, state[1])
        elif action == 1:  # down
            return (state[0] + 1, state[1])
        elif action == 2:  # left
            return (state[0], state[1] - 1)
        elif action == 3:  # right
            return (state[0], state[1] + 1)
        return state

    P = np.zeros((n_states, n_actions, n_states))
    R = np.zeros((n_states, n_actions))

    for s in range(n_states):
        state = index_to_state(s)
        for a in range(n_actions):
            if state == state_A:
                next_state = state_A_prime
                reward = 10
            elif state == state_B:
                next_state = state_B_prime
                reward = 5
            else:
                next_state = move(state, a)
                if not is_valid(next_state):
                    next_state = state
                    reward = -1
                else:
                    reward = 0
            s_next = state_to_index(next_state)
            P[s, a, s_next] = 1.0
            R[s, a] = reward

    gamma = 0.9
    return MDP(S, A, P, R, gamma)


def test_value_iteration_gridworld():
    mdp = create_gridworld_mdp()
    V, policy = value_iteration(mdp, tol=1e-10)

    # print results with .2f formatting
    np.set_printoptions(precision=1, suppress=True)
    print("Optimal Value Function:")
    print(V.reshape((5, 5)))


if __name__ == "__main__":
    test_value_iteration_gridworld()
