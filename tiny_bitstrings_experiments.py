"""
This module executed hyper-parameter experiments on the Tiny BitString MDP.
The experiments are done using all three learning methods
"""

import time
import matplotlib.pyplot as plt

from value_iterations import ValueIteration
from policy_iteration import PolicyIteration
from q_learning import QLearning
from mdp import BitStrings

print("\nExperiments on Tiny Bit Strings")
print("- - - - - - - - - - - - - - - -")

# Create the MDP object
tiny_bitstrings = BitStrings(size=8)

# Time of exetution and last state-value for different trained
results = {}

# Experiements using Value Iterations using different hyper-parameters
print("- Using value iteration")
begin_tstamp = time.time()
_, mean_state_values = ValueIteration(tiny_bitstrings, gamma=1, epsilon=0.0001)()
end_tstamp = time.time()
results["Value Iteration"] = [end_tstamp - begin_tstamp, mean_state_values[-1]]

# Experiements using Policy Iterations using different hyper-parameters
print("- Using policy iterations")
begin_tstamp = time.time()
_, _, mean_state_values = PolicyIteration(tiny_bitstrings, gamma=0.9, epsilon=0.0001)()
end_tstamp = time.time()
results["Policy Iteration"] = [end_tstamp - begin_tstamp, mean_state_values[-1]]

# Experiements using Q-learning using different hyper-parameters
print("- Using Q learning")
begin_tstamp = time.time()
_, mean_state_values = QLearning(tiny_bitstrings, alpha=0.01, eps=1)(num_episodes=35000)
end_tstamp = time.time()
results["Q Learning"] = [end_tstamp - begin_tstamp, mean_state_values[-1]]

# Print the results
print("\n- Results")
print("\tAlgorithm      Execution time     V(s)")
for key, value in results.items():
    print(f"\t{key}     {value[0]:.5f}     {value[1]:.2f}")
