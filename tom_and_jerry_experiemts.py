"""
This module executed hyper-parameter experiments on the Tom & Jerry MDP.
The experiments are done using all three learning methods
"""

import time
import matplotlib.pyplot as plt

from value_iterations import ValueIteration
from policy_iteration import PolicyIteration
from q_learning import QLearning
from mdp import TomAndJerry

print("\nExperiments on Tom and Jerry")
print("- - - - - - - - - - - - - -")

# Create the MDP object
tom_and_jerry = TomAndJerry()

# Colors used for plotting
colors = ['blue', 'green', 'black', 'pink', 'orange', 'blue', 'green', 'black', 'pink']
linestyles = ["-", "-", "-", "-", "-", "--", "--", "--", "--"]

# Experiements using Value Iterations using different hyper-parameters
print("\n- Using value iterations")
fig = plt.figure(figsize=(10,8))
ax = fig.add_subplot(1,1,1)
grid_search = [(0.1, 0.0001), (0.9, 0.0001), (1, 0.0001), (1, 0.1)]  # List of hyper-param (gamma, eps)
vi_results = {}   # Time of exetution and last state-value for different hyperparameter value
optimum_policy = None
best_state_value = 0
c_i = 0
for gamma, eps in grid_search:
    begin_tstamp = time.time()
    value_iter = ValueIteration(tom_and_jerry, gamma=gamma, epsilon=eps)
    trained_policy, mean_state_values = value_iter()
    ax.plot(mean_state_values, color=colors[c_i], linestyle=linestyles[c_i], label=f"Gamma = {gamma}, Epsilon = {eps}")
    end_tstamp = time.time()
    vi_results[(gamma, eps)] = [end_tstamp - begin_tstamp, mean_state_values[-1]]
    if mean_state_values[-1] > best_state_value:
        optimum_policy = trained_policy
        best_state_value = mean_state_values[-1]
    c_i += 1
ax.set_xlabel("Number of iterations")
ax.set_ylabel("Average state value V(s)")
ax.set_title("Average state value VS No. of iterations for Value Iterations")
ax.legend()
fig.savefig("./plots/tom_and_jerry_value_iterations.png")
plt.close(fig)
print("  - Results")
print("\tGamma\tEpsilon\tExecution time\tV(s)")
for key, value in vi_results.items():
    print(f"\t{key[0]}\t{key[1]}\t{value[0]:.5f}\t\t{value[1]:.2f}")
print("  - Optimum policy")
for r in range(4):
    print("\t", end="")
    for c in range(4):
        if (r,c) not in optimum_policy:
            print("N", end=" ")
        else:
            print(optimum_policy[(r,c)], end=" ")
    print()
print()

# Experiements using Policy Iterations using different hyper-parameters
print("\n- Using policy iterations")
fig1 = plt.figure(figsize=(10,8))
ax1 = fig1.add_subplot(1,1,1)
fig2 = plt.figure(figsize=(10,8))
ax2 = fig2.add_subplot(1,1,1)
grid_search = [(0.1, 0.0001), (0.9, 0.0001), (1, 0.0001), (1, 0.1)]  # List of hyper-param (gamma, eps)
pi_results = {}   # Time of exetution and last state-value for different hyperparameter value
optimum_policy = None
best_state_value = 0
c_i = 0
for gamma, eps in grid_search:
    begin_tstamp = time.time()
    policy_iter = PolicyIteration(tom_and_jerry, gamma=gamma, epsilon=eps)
    trained_policy, mean_policy_changes, mean_state_values = policy_iter()
    ax1.plot(mean_state_values, color=colors[c_i], linestyle=linestyles[c_i], label=f"Gamma = {gamma}, Epsilon = {eps}")
    ax2.plot(mean_policy_changes, color=colors[c_i], linestyle=linestyles[c_i], label=f"Gamma = {gamma}, Epsilon = {eps}")
    end_tstamp = time.time()
    pi_results[(gamma, eps)] = [end_tstamp - begin_tstamp, mean_state_values[-1]]
    if mean_state_values[-1] > best_state_value:
        optimum_policy = trained_policy
        best_state_value = mean_state_values[-1]
    c_i += 1
ax1.set_xlabel("Number of iterations")
ax1.set_ylabel("Average state value V(s)")
ax1.set_title("Average state value VS No. of iterations for Policy Iterations")
ax1.legend()
fig1.savefig("./plots/tom_and_jerry_policy_iterations_V.png")
plt.close(fig1)
ax2.set_xlabel("Number of iterations")
ax2.set_ylabel("Mean policy changes")
ax2.set_title("Mean policy changes VS No. of iterations for Policy Iterations")
ax2.legend()
fig2.savefig("./plots/tom_and_jerry_policy_iterations_policy_changes.png")
plt.close(fig2)
print("  - Results")
print("\tGamma\tEpsilon\tExecution time\tV(s)")
for key, value in pi_results.items():
    print(f"\t{key[0]}\t{key[1]}\t{value[0]:.5f}\t\t{value[1]:.2f}")
print("  - Optimum policy")
for r in range(4):
    print("\t", end="")
    for c in range(4):
        if (r,c) not in optimum_policy:
            print("N", end=" ")
        else:
            print(optimum_policy[(r,c)], end=" ")
    print()
print()

# Experiements using Q-learning using different hyper-parameters
print("\n- Using Q learning")
fig = plt.figure(figsize=(10,8))
ax = fig.add_subplot(1,1,1)
grid_search = [(0.001, 1), (0.01, 1), (0.01, 0.5), (0.001, 0.5)]  # List of hyper-param (gamma, eps)
ql_results = {}   # Time of exetution and last state-value for different hyperparameter value
optimum_policy = None
best_state_value = 0
c_i = 0
for alpha, eps in grid_search:
    begin_tstamp = time.time()
    q_learning = QLearning(tom_and_jerry, alpha=alpha, eps=eps)
    trained_policy, mean_state_values = q_learning()
    ax.plot(mean_state_values, color=colors[c_i], linestyle=linestyles[c_i], label=f"Alpha = {alpha}, Epsilon = {eps}")
    end_tstamp = time.time()
    ql_results[(alpha, eps)] = [end_tstamp - begin_tstamp, mean_state_values[-1]]
    if mean_state_values[-1] > best_state_value:
        optimum_policy = trained_policy
        best_state_value = mean_state_values[-1]
    c_i += 1
ax.set_xlabel("Number of iterations")
ax.set_ylabel("Average state value V(s)")
ax.set_title("Average state value VS No. of iterations for Policy Iterations")
ax.legend()
fig.savefig("./plots/tom_and_jerry_q_learning.png")
plt.close(fig)
print("  - Results")
print("\tAlpha\tEpsilon\tExecution time\tV(s)")
for key, value in ql_results.items():
    print(f"\t{key[0]}\t{key[1]}\t{value[0]:.5f}\t\t{value[1]:.2f}")
print("  - Optimum policy")
for r in range(4):
    print("\t", end="")
    for c in range(4):
        if (r,c) not in optimum_policy:
            print("N", end=" ")
        else:
            print(optimum_policy[(r,c)], end=" ")
    print()
print()
