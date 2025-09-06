import numpy as np


class MDP:
    def __init__(
        self,
        S: np.ndarray,
        A: np.ndarray,
        P: np.ndarray,
        R: np.ndarray,
        gamma: float = 0.995,
    ):
        """
        Markov Decision Process (MDP) class.

        Parameters
        ----------
        S : np.ndarray
            State space of shape (n_state, state_dim).
        A : np.ndarray
            Action space of shape (n_action, action_dim).
        P : np.ndarray
            Transition probability matrix of shape (n_state, n_action, n_state).
        R : np.ndarray
            Reward matrix of shape (n_state, n_action).
        gamma : float, optional
            Discount factor, by default 0.995.
        """
        self.S = S
        self.A = A
        self.P = P
        self.R = R
        self.gamma = gamma
