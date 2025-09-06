from ..mdp import MDP
import numpy as np


def value_iteration(mdp: MDP, tol=1e-6):
    """
    Value iteration algorithm for solving a Markov Decision Process (MDP).

    Parameters
    ----------
    mdp : MDP
        The Markov Decision Process to solve.
    tol : float, optional
        Tolerance for stopping criterion, by default 1e-6.

    Returns
    -------
    V : np.ndarray
        Optimal value function.
    policy : np.ndarray
        Optimal policy.
    """
    S = mdp.S  # State space
    A = mdp.A  # Action space
    P = mdp.P  # Transition probabilities
    R = mdp.R  # Reward matrix
    gamma = mdp.gamma  # Discount factor

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
    for state in range(n_state):
        policy[state] = np.argmax(R[state] + gamma * np.dot(P[state], V))

    # get dictionary policy, value
    policy_dict = {tuple(s): a for s, a in zip(S, policy)}
    V_dict = {tuple(s): v for s, v in zip(S, V)}

    return V, policy
