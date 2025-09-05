from ..mdp import MDP
import numpy as np

def value_iteration(P, R, gamma=0.995, tol=1e-6)
def value_iteration(P, R, gamma=0.995, tol=1e-6):
    n_state, n_action = R.shape
    V = np.zeros(n_state)
    delta = np.inf
    k = 0
    while delta > tol:
        V_all = R + gamma * np.dot(P, V)
        V_new = np.max(V_all, axis=1)
        delta = np.max(np.abs(V_new - V))
        V = V_new
        k += 1

    # compute the optimal policy
    policy = np.zeros(n_state, dtype=int)
    for i in range(n_state):
        policy[i] = np.argmax(R[i] + gamma * np.dot(P[i], V))

    return V, policy
