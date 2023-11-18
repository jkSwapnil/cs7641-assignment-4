# This module implements the policy iteration

import random


class PolicyIteration:

    """ Finds optimum policy using Policy Iteration """

    def __init__(self, mdp, gamma=0.9, epsilon=0.0001):
        """ Constructor
        Parameters:
            - mdp: Markov devision process object
            - gamma: Discount factor
            - epsilon: The threshold to stop the value iterations
        """
        self.mdp = mdp    # MDP to train policy for
        self.gamma = gamma
        self.eps = epsilon
        self.allowed_states, self.allowed_actions = self.mdp.get_allowed_states_and_actions()
        self.V = {}       # Value function
        self.policy = {}  # Policy
        # Initialize a random policy
        for s in self.allowed_states:
            self.policy[s] = random.choice(self.allowed_actions)

    def __Q(self, s, a):
        """ Evaluate optimum Q(s,a)
        Paramaters:
            - s: Current state
            - a: Current action
        Returns:
            Optimum Q(s,a)
        """
        Q_s_a = 0
        for s_prime in self.allowed_states:
            Q_s_a += self.mdp.get_transition_prob(s, a, s_prime) * (self.mdp.get_reward(s, a, s_prime) + self.gamma * self.V[s_prime])
        return Q_s_a
    
    def __evaluate_policy(self):
        """ Estimate the on policy value function for the current policy """
        # Initialize the value function to 0
        for s in self.allowed_states:
            self.V[s] = 0
        # Update using the Bellamn equation till convergence
        while True:
            delta = 0
            for s in self.allowed_states:
                old_v = self.V[s]
                self.V[s] = self.__Q(s, self.policy[s])
                delta = max(delta, abs(old_v - self.V[s]))
            if delta < self.eps:
                break
        return

    def __policy_update(self):
        """ Update policy based on the current state values
        Returns:
            Mean of action changes for all the states
        """
        total_policy_change = 0   # Sum of | old_action - new_action | for each state
        for s in self.allowed_states:
            old_a = self.policy[s]
            Q_s = [self.__Q(s,a) for a in self.allowed_actions]
            self.policy[s] = Q_s.index(max(Q_s))  # argmax_a[ Q(s,a) ]
            total_policy_change += abs(old_a - self.policy[s])
        return total_policy_change/len(self.allowed_states)

    def __call__(self):
        """ Execute the policy iteration using Bellman Equation
        Returns:
            - Optimum policy obtained using policy iteration
            - Mean change in the policy
            - State value for optimum policy
        """
        mean_policy_changes = []
        while True:
            # Policy evaluation
            self.__evaluate_policy()
            # Policy update
            mean_change = self.__policy_update()
            mean_policy_changes.append(mean_change)
            if mean_change == 0:
                break
        return self.policy, mean_policy_changes, self.V


if __name__ == "__main__":

    from mdp import TomAndJerry, BitStrings

    # Create the Tom and Jerry MDP
    # Execute the policy iteration
    print("\nTest policy iteration for Tom and Jerry MDP")
    print("- - - - - - - - - - - - - - - - - - - - - ")
    tom_and_jerry = TomAndJerry()
    policy_iter = PolicyIteration(tom_and_jerry)
    optimum_policy, _, _ = policy_iter()
    print("- Done")

    # Create the Bit Strings MDP
    # Execute the policy iteration
    print("\nTest policy iteration for Bit Strings MDP")
    print("- - - - - - - - - - - - - - - - - - - -")
    bit_strings = BitStrings()
    policy_iter = PolicyIteration(bit_strings)
    optimum_policy, _, _ = policy_iter()
    print("- Done")
    print()
