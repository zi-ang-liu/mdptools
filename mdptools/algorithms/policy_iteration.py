from ..mdp import MDP


def policy_iteration(mdp: MDP, tol=1e-6, max_iter=1000):
    pass

    # def policy_evaluation(self, policy, start_value=None):
    #     """
    #     Policy evaluation function

    #     Args:
    #     policy: list
    #         Policy to be evaluated

    #     Returns:
    #     current_pi_v_df: pandas.DataFrame
    #         Dataframe of the current policy and value function
    #     """
    #     if start_value is not None:
    #         value = start_value.copy()
    #     else:
    #         value = np.zeros(len(self.state_space))

    #     delta = np.inf
    #     k = 0
    #     while delta >= self.theta:
    #         k = k + 1
    #         value_old = value.copy()
    #         for i, state in enumerate(self.state_space):
    #             action = policy[i]
    #             value[self.state_dict[state]] = sum(
    #                 [
    #                     prob
    #                     * (reward + self.gamma * value_old[self.state_dict[next_state]])
    #                     for (next_state, reward), prob in self.dynamics[
    #                         state, action
    #                     ].items()
    #                 ]
    #             )
    #         # check RuntimeWarning: invalid value encountered in subtract delta = np.max(np.abs(value - value_old))
    #         # if np.isnan(value).any():
    #         #     print('Nan value encountered in value function')
    #         #     break
    #         # elif np.isinf(value).any():
    #         #     print('Inf value encountered in value function')
    #         #     break
    #         delta = np.max(np.abs(value - value_old))
    #         if self.verbose:
    #             print("Iteration {}, delta = {}".format(k, delta))

    #     df_value = pd.DataFrame(value, columns=self.value_name)
    #     df_state = pd.DataFrame(self.state_space, columns=self.state_name)
    #     df_action = pd.DataFrame(policy, columns=self.action_name)
    #     current_pi_v_df = pd.concat([df_state, df_action, df_value], axis=1)
    #     return current_pi_v_df

    # def policy_improvement(self, value, policy):
    #     """
    #     Policy improvement function

    #     Args:
    #     value: list
    #         Value function
    #     policy: list
    #         Policy to be improved

    #     Returns:
    #     new_policy: list
    #         Improved policy
    #     stable: bool
    #         Whether the policy is stable
    #     """
    #     stable = True
    #     for i, state in enumerate(self.state_space):
    #         old_action = policy[i]
    #         q_max_value = -np.inf
    #         for action in self.action_space:
    #             q_value_temp = sum(
    #                 [
    #                     prob
    #                     * (reward + self.gamma * value[self.state_dict[next_state]])
    #                     for (next_state, reward), prob in self.dynamics[
    #                         state, action
    #                     ].items()
    #                 ]
    #             )
    #             if q_value_temp > q_max_value:
    #                 q_max_value = q_value_temp
    #                 policy[i] = action
    #         if old_action != policy[i]:
    #             stable = False
    #     return policy, stable

    # def policy_iteration(self, start_value=None, start_policy=None):
    #     """
    #     Policy iteration function

    #     Returns:
    #     opt_pi_v_df: pandas.DataFrame
    #         Dataframe of the optimal policy and value function
    #     """
    #     if start_value is not None:
    #         value = start_value.copy()
    #         policy = start_policy.copy()
    #     else:
    #         policy = np.zeros(len(self.state_space), dtype=int)
    #         value = np.zeros(len(self.state_space))

    #     stable = False
    #     k = 0
    #     while not stable:
    #         k = k + 1
    #         current_pi_v_df = self.policy_evaluation(policy, value)
    #         value = current_pi_v_df[self.value_name].values
    #         policy, stable = self.policy_improvement(value, policy)
    #         if self.verbose:
    #             print("Iteration {}, stable = {}".format(k, stable))
    #     df_state = pd.DataFrame(self.state_space, columns=self.state_name)
    #     df_action = pd.DataFrame(policy, columns=self.action_name)
    #     df_value = pd.DataFrame(value, columns=self.value_name)
    #     opt_pi_v_df = pd.concat([df_state, df_action, df_value], axis=1)
    #     return opt_pi_v_df
