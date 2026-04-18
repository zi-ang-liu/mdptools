from ..mdp import MDP
import numpy as np


def policy_iteration(mdp: MDP, tol=1e-6, max_iter=1000):
    """
    Policy iteration algorithm for solving a Markov Decision Process (MDP).

    Parameters
    ----------
    mdp : MDP
        The Markov Decision Process to solve.
    tol : float, optional
        Tolerance for policy-evaluation convergence, by default 1e-6.
    max_iter : int, optional
        Maximum number of policy-iteration and policy-evaluation steps,
        by default 1000.

    Returns
    -------
    V : np.ndarray
        Optimal value function.
    policy : np.ndarray
        Optimal policy.
    """
    P = mdp.P
    R = mdp.R
    gamma = mdp.gamma

    n_state, _ = R.shape
    states = np.arange(n_state)

    # Start from a deterministic default policy and zero values.
    policy = np.zeros(n_state, dtype=int)
    V = np.zeros(n_state)
    V_new = np.empty_like(V)
    q_values = np.empty_like(R)

    for _ in range(max_iter):
        # Policy evaluation: iteratively solve V^pi.
        P_pi = P[states, policy]
        R_pi = R[states, policy]

        for _ in range(max_iter):
            np.dot(P_pi, V, out=V_new)
            V_new *= gamma
            V_new += R_pi

            delta = np.max(np.abs(V_new - V))
            V, V_new = V_new, V
            if delta <= tol:
                break

        # Policy improvement: greedify with respect to the current values.
        np.dot(P, V, out=q_values)
        q_values *= gamma
        q_values += R
        new_policy = np.argmax(q_values, axis=1)

        if np.array_equal(new_policy, policy):
            break

        policy = new_policy

    return V, policy
