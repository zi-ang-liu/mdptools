from ..mdp import MDP
import numpy as np


def value_iteration(mdp: MDP, tol=1e-6, max_iter=10_000):
    """
    Value iteration algorithm for solving a Markov Decision Process (MDP).

    Parameters
    ----------
    mdp : MDP
        The Markov Decision Process to solve.
    tol : float, optional
        Tolerance for stopping criterion, by default 1e-6.
    max_iter : int, optional
        Maximum number of iterations as a safety guard, by default 10_000.

    Returns
    -------
    V : np.ndarray
        Optimal value function.
    policy : np.ndarray
        Optimal policy.

    Algorithm
    ---------
    Initialize V(s) = 0 for all states s.
    Get tol, max_iter from arguments.
    Loop until delta < tol or max_iter is reached:
        For each state:
            v = V(s)
            V(s) = max_a [ R(s, a) + gamma * sum_{s'} P(s' | s, a) * V(s') ]
            delta = max(delta, |v - V(s)|)

    Get the optimal policy:
    For each state:
        policy(s) = argmax_a [ R(s, a) + gamma * sum_{s'} P(s' | s, a) * V(s') ]
    """

    # Read once to local variables (faster attribute access in tight loops).
    P = mdp.P  # (n_state, n_action, n_state)
    R = mdp.R  # (n_state, n_action)
    gamma = mdp.gamma

    n_state, n_action = R.shape
    V = np.zeros(n_state)
    # Reusable buffers to avoid per-iteration allocations.
    V_new = np.empty_like(V)
    q_values = np.empty((n_state, n_action), dtype=V.dtype)
    diff = np.empty_like(V)
    delta = np.inf

    k = 0
    while delta > tol and k < max_iter:
        # Bellman backup for all states/actions: Q(s, a) = R(s, a) + gamma * E[V(s')].
        np.dot(P, V, out=q_values)
        q_values *= gamma
        q_values += R

        # V_new(s) = max_a Q(s, a) and delta = max_s |V_new(s) - V(s)|.
        np.max(q_values, axis=1, out=V_new)
        np.subtract(V_new, V, out=diff)
        np.abs(diff, out=diff)
        delta = np.max(diff)

        # Swap buffers to avoid allocating a new V at each iteration.
        V, V_new = V_new, V
        k += 1

    # Compute the optimal policy from the converged value function.
    np.dot(P, V, out=q_values)
    q_values *= gamma
    q_values += R
    policy = np.argmax(q_values, axis=1)

    return V, policy
