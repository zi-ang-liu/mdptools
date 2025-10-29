# MDP Tools

MDP Tools is a Python library for solving Markov Decision Processes (MDPs). 


## Installation

You can install the package using pip:

```bash
git clone https://github.com/zi-ang-liu/mdptools.git
cd mdptools
pip install .
```

## MDP Class

The `MDP` class represents a Markov Decision Process with the following parameters:

- `S`: State space of shape (n_states, state_dim).
- `A`: Action space of shape (n_actions, action_dim).
- `P`: Transition probability matrix of shape (n_actions, n_states, n_states).
- `R`: Reward matrix of shape (n_states, n_actions).
- `gamma`: Discount factor (default is 0.995).

## Algorithms

### Value Iteration 

The `value_iteration` function implements the Value Iteration algorithm to compute the optimal value function and policy for a given MDP. It takes the following parameters:
- `mdp`: An instance of the `MDP` class.
- `theta`: A small threshold for determining the accuracy of estimation (default is 1e-6).

It returns:
- `V`: Optimal value function.
- `policy`: Optimal policy.

### Todo

- [ ] Add Policy Iteration algorithm.
- [ ] Add Q-Learning algorithm.
- [ ] Modify the simulation function.

## Example Usage

### Basic Usage

```python
S = np.array([[0], [1]])
A = np.array([[0], [1]])
P = np.array([[[1.0, 0.0], [0.0, 1.0]],
              [[0.0, 1.0], [1.0, 0.0]]])
R = np.array([[1.0, 5.0],
              [3.0, 2.0]])
gamma = 0.9

mdp = MDP(S, A, P, R, gamma)
V, policy = value_iteration(mdp)
print("Optimal Policy:\n", policy)
print("Optimal Value Function:\n", V)
```


### Verified Example

Please refer to the `tests` folder for verified examples.

- Sutton et al. (2018) Example 3.8, p. 65