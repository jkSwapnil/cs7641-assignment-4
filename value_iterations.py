# This module implements the value iteration


class ValueIteration:

    """ Finds optimum policy using value iteration """

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
        self.V = {}       # Value function
        self.policy = {}  # Trained policy
        # Initialize the state values to 0
        self.allowed_states, self.allowed_actions = self.mdp.get_allowed_states_and_actions()
        for s in self.allowed_states:
            self.V[s] = 0

    def __get_mean_V(self):
        """ Evaluate mean state-value across all states
        Returns:
            mean V (int)
        """
        return sum(self.V.values())/len(self.V)

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

    def __call__(self):
        """ Execute the value iteration using Bellman Equation
        Returns:
            - Optimum policy obtained using value iteration
            - Mean state values as the iteration processes
        """
        # Update using Bellman Equation till convergence
        mean_state_values = [self.__get_mean_V()]  # Average state-values as the iteration progresses
        while True:
            delta = 0
            for s in self.allowed_states:
                old_v = self.V[s]
                self.V[s] = max([self.__Q(s, a) for a in self.allowed_actions])
                delta = max(delta, abs(old_v - self.V[s]))
            mean_state_values.append(self.__get_mean_V())
            if delta < self.eps:
                break
        # Evaluate the optimum policy
        for s in self.allowed_states:
            Q_s = [self.__Q(s,a) for a in self.allowed_actions]
            self.policy[s] = Q_s.index(max(Q_s))
        return self.policy, mean_state_values


if __name__ == "__main__":

    from mdp import TomAndJerry, BitStrings

    # Create the Tom and Jerry MDP
    # Execute the value iteration
    print("\nTest value iteration for Tom and Jerry MDP")
    print("- - - - - - - - - - - - - - - - - - - - -")
    tom_and_jerry = TomAndJerry()
    value_iter = ValueIteration(tom_and_jerry)
    optimum_policy, _ = value_iter()
    print("- Done")

    # Create the Bit Strings MDP
    # Execute the value iteration
    print("\nTest value iteration for Bit Strings MDP")
    print("- - - - - - - - - - - - - - - - - - - -")
    bit_strings = BitStrings()
    value_iter = ValueIteration(bit_strings)
    optimum_policy, _ = value_iter()
    print("- Done")
    print()
