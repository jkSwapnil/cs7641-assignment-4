# This module implements Q-learning

import random
from tqdm import tqdm


class QLearning:

    """ Find optimum policy using Q-Learning """

    def __init__(self, mdp, alpha=0.01, gamma=0.9, eps=1):
        """ Constructor
        Parameters:
            - mdp: Markov Decision Process object
            - alpha: learning rate (float)
            - gamma: discount factor (float)
            - eps: expl
            pass
        """
        self.mdp = mdp
        self.alpha = alpha
        self.gamma = gamma
        self.eps = eps
        self.Q = {}
        # Initialize the Q-values to 0
        self.allowed_states, self.allowed_actions = self.mdp.get_allowed_states_and_actions()
        for s in self.allowed_states:
            self.Q[s] = [0]*len(self.allowed_actions)

    def __select_action(self, s):
        """ Select action using epsilon-greedy
        Parameters:
            - s: State for which to select the action
        Returns:
            - Selected action
        """
        p = random.random()
        if p < self.eps:  # Return a randomly selected action
            return random.randint(0, len(self.allowed_actions) - 1)
        else:  # Return the optimum action
            return self.Q[s].index(max(self.Q[s]))

    def __get_V(self):
        """ Compute optimum V from the optimum Q
        Returns:
            - Value function
        """
        V = {}
        for s in self.allowed_states:
            V[s] = max(self.Q[s])
        return V

    def __call__(self, num_episodes=10000):
        """ Execute Q-Learning for `num_episodes' episodes
        Parameters:
            - num_episodes: Number of episodes to run Q-Learning for (int)
        Returns:
            - optimum learnt policy
        """
        mean_state_values = [sum(self.__get_V().values())/len(self.allowed_states)]
        for episode in tqdm(range(num_episodes)):
            # print(episode)
            self.eps = 0.9995 * self.eps  # Reduce the exploration is a step-wise manner
            s, _, episode_end = self.mdp.reset()  # Reset the MDP
            while not episode_end:
                a = self.__select_action(s)  # Select the action in epsilon-greedy fashion
                s_prime, r, episode_end = self.mdp.step(a)  # Execute the MDP
                self.Q[s][a] = self.Q[s][a] + self.alpha * (r + self.gamma * max(self.Q[s_prime]) - self.Q[s][a])
                s = s_prime  # Update as current state
                mean_state_values.append(sum(self.__get_V().values())/len(self.allowed_states))
        # Create the optimum policy
        policy = {}
        for s in self.allowed_states:
            policy[s] = self.Q[s].index(max(self.Q[s]))

        return policy, mean_state_values


if __name__ == "__main__":

    from mdp import TomAndJerry, BitStrings

    # Create the Tom and Jerry MDP
    # Execute the Q-Learning
    print("\nTest Q-Learning for Tom and Jerry MDP")
    print("- - - - - - - - - - - - - - - - - - ")
    tom_and_jerry = TomAndJerry()
    q_learning = QLearning(tom_and_jerry)
    optimum_policy, _ = q_learning()
    print("- Done")

    # Create the Bit Strings MDP
    # Execute the Q-Leaning
    print("\nTest Q-Learning for Bit Strings MDP")
    print("- - - - - - - - - - - - - - - - - -")
    bit_strings = BitStrings()
    q_learning = QLearning(bit_strings)
    optimum_policy, _ = q_learning()
    print("- Done")
    print()
