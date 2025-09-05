class MDP:
    def __init__(
        self, states, actions, transition_probabilities, rewards, discount_factor=1.0
    ):
        """
        Initialize a Markov Decision Process (MDP).

        :param states: A list of states in the MDP.
        :param actions: A list of actions available in the MDP.
        :param transition_probabilities: A dictionary mapping (state, action) pairs to a list of (next_state, probability) tuples.
        :param rewards: A dictionary mapping (state, action) pairs to rewards.
        :param discount_factor: The discount factor for future rewards (default is 1.0).
        """
        self.states = states
        self.actions = actions
        self.transition_probabilities = transition_probabilities
        self.rewards = rewards
        self.discount_factor = discount_factor
