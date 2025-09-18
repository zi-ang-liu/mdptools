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

### Simple usage

```python
import numpy as np
from mdptools.mdp import MDP
from mdptools.algorithms.value_iteration import value_iteration

P = np.array([[[0.8, 0.2], [0.1, 0.9]], [[0.7, 0.3], [0.4, 0.6]]])
R = np.array([[5, 10], [0, 2]])
mdp = MDP(P, R, gamma=0.9)
V, policy = value_iteration(mdp)
print("Optimal Value Function:", V)
print("Optimal Policy:", policy)
```

### Verified Example

Currently, the following examples are implemented and verified:

- Sutton et al. (2018) Example 3.8, p. 65