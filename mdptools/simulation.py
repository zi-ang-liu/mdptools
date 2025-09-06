def simulation(self, policy, start_state, n_steps=1000):
    """
    Simulation function

    Args:
    policy: list
        Policy to be evaluated
    start_state: int
        Starting state
    n_steps: int
        Number of steps to simulate

    Returns:
    state_sequence: list
        Sequence of states
    action_sequence: list
        Sequence of actions
    reward_sequence: list
        Sequence of rewards
    """
    state_sequence = [start_state]
    action_sequence = []
    reward_sequence = []
    state = start_state
    for _ in range(n_steps):
        action = policy[self.state_dict[state]]
        next_state, reward = self.dynamics[state, action].sample()
        state_sequence.append(next_state)
        action_sequence.append(action)
        reward_sequence.append(reward)
        state = next_state
    return state_sequence, action_sequence, reward_sequence
