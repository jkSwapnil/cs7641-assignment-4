# This module defines the 2 MDPs used in this project

import random


class MDP:
    """ Interface for the MDPs """

    def get_allowed_states_and_actions(self):
        """ Return all possible states and actions in the MDP 
        Returns:
            List[allowed_states], List[allowed_actions]
        """
        pass

    def get_terminal_states(self):
        """ Return all the terminal states """
        pass

    def get_transition_prob(self, state, action, next_state):
        """ Get transition probability for (state, action, next_state)
        Parameters:
            state: current state
            action: current action (int)
            next_state: next state
        Returns:
            p: transition probability (float)
        """
        pass

    def get_reward(self, state, action, next_state):
        """ Get reward for (state, action, next_state) transition
        Parameters:
            state: current state
            action: current action (int)
            next_state: next state
        Returns:
            p: transition probability (float)
        """
        pass
        
    def reset(self):
        """ Reset the MDP
        Returns: initial_state, reward(int), episode_end(False)
        """
        pass

    def step(self, action):
        """ Update the MDP based on the selected action
        Parameters:
            action: selected action (int)
        Returns:
            next_state, reward (int), episode_ended (boo.)
        """
        pass


class TomAndJerry(MDP):
    """ Concrete MDP for Tom and Jerry game 
    
    Jerry has to reach Cheeze avoiding the traps
    - Jerry stats from (0,0)
    - Rewards:
        +1 reward if Jerry reaches cheeze
        -1 reward if Jerry reaches Tom
    - Actions: 
        Jerry can move top(0), right(1), down(2), left(3)
    - Environment transition: 
        Jerry moves in the desired direction with probability 1/3
        Jerry moves perpendicular with probability 1/3
        If there is a trap/grid-border in any direction, the movement does not happen
    """
    
    def __init__(self):
        self.jerry = (0, 0)  # Current position of Jerry
        self.tom = (1,3)     # Position of Tom
        self.cheeze = (2,3)  # Position of Cheeze
        self.traps = set([(2,2), (3,0)])       # Position of traps
        self.allowed_actions = set([0,1,2,3])  # Allowed actions

    def get_allowed_states_and_actions(self):
        # Get all the allowed states
        allowed_states = []
        for r in range(4):
            for c in range(4):
                if (r,c) not in self.traps:
                    allowed_states.append((r,c))
        # Get the allowed actions
        allowed_actions = list(self.allowed_actions)
        return allowed_states, allowed_actions
    
    def get_terminal_states(self):
        return [self.tom, self.cheeze]

    def get_transition_prob(self, state, action, next_state):
        # If the state is out of bonds
        if state[0] < 0 or state[0] > 3 or state[1] < 0 or state[1] > 3:
            return 0
        # If action is out of bonds
        if action not in self.allowed_actions:
            return 0
        # If the state is terminal
        if state in self.get_terminal_states():
            return 0
        # Get the possible next_states from the current state 
        possible_next_states = set()
        dx = [-1, 0, 1, 0]
        dy = [0, 1, 0, -1]
        possible_actions = [1,3] if action%2 == 0 else [0,2]
        possible_actions += [action]
        for a in possible_actions:
            new_x = state[0] + dx[a]
            new_y = state[1] + dy[a]
            if (new_x, new_y) not in self.traps and 0 <= new_x <= 3 and 0 <= new_y <= 3:
                possible_next_states.add((new_x, new_y))
        if next_state in possible_next_states:
            return 1/3
        return 0

    def get_reward(self, state, action, next_state):
        if next_state == self.tom:
            return -1
        elif next_state == self.cheeze:
            return 1
        else:
            return 0

    def reset(self):
        self.jerry = (0,0)
        return self.jerry, 0, False
    
    def step(self, action):
        # Raise expection if the action is invalid
        if action not in self.allowed_actions:
            raise Exception(f"'{action}' is not a valid action")
        # Do nothing if the game has ended
        if self.jerry in self.get_terminal_states():
            return self.jerry, 0, True
        # Update Jerry's position using the transition model
        if action%2 == 0:
            action = random.choice([action] + [1,3])
        else:
            action = random.choice([action] + [0,2])
        dx = [-1, 0, 1, 0]
        dy = [0, 1, 0, -1]
        new_x = self.jerry[0] + dx[action]
        new_y = self.jerry[1] + dy[action]
        reward = 0
        episode_end = False
        if (new_x, new_y) not in self.traps and 0 <= new_x <= 3 and 0 <= new_y <= 3:
            reward += self.get_reward(self.jerry, action, (new_x, new_y))
            self.jerry = (new_x, new_y)
            if reward != 0:
                episode_end = True  
        return self.jerry, reward, episode_end
        

class BitStrings(MDP):
    """ Concrete MDP for Bit Strings of size 9

    9 size of strings
    - Start from '000000000'
    - Rewards:
        +1 reward for '010101010' or '101010101'
        -2 reward for '111111111'
    - Actions:
        Any of the 9 bits can be flipped
    - Environment transition: 
        If an action is taken to flip bit i, then with a 0.5 chance i+1 can can be flipped instead
        For the last bit, it is guranteed to flip as there is no next bit to it
    """

    def __init__(self):
        self.state = "000000000"
        self.positive_terminal_states = ["010101010", "101010101"]
        self.negative_terminal_states = ["111111111"]

    def __generate_allowed_states(self, state, allowed_states):
        if len(state) == 9:
            allowed_states.append(state)
            return
        self.__generate_allowed_states(state + '1', allowed_states)
        self.__generate_allowed_states(state + '0', allowed_states)
        return

    def get_allowed_states_and_actions(self):
        # Get all the allowed states
        allowed_states = []
        self.__generate_allowed_states("", allowed_states)
        # Get the allowed actions
        allowed_actions = list(range(9))
        return allowed_states, allowed_actions

    def get_terminal_states(self):
        return self.positive_terminal_states + self.negative_terminal_states

    def get_transition_prob(self, state, action, next_state):
        # If action is out of bonds
        if action < 0 or action > 8:
            return 0
        # If the state is terminal
        if state in self.get_terminal_states():
            return 0
        # Get the possible next_states from the current state
        possible_next_states = set()
        new_bit_value = 1 - int(state[action])
        possible_next_states.add(state[0:action] + str(new_bit_value) + state[action+1:])
        if action < 8:
            action = action + 1
            new_bit_value = 1 - int(state[action])
            possible_next_states.add(state[0:action] + str(new_bit_value) + state[action+1:])
        # Return the probability
        if next_state in possible_next_states:
            return 0.5
        return 0

    def get_reward(self, state, action, next_state):
        if next_state in self.negative_terminal_states :
            return -2
        elif next_state in self.positive_terminal_states:
            return 1
        else:
            return 0

    def reset(self):
        self.state = "000000000"
        return self.state, 0, False
    
    def step(self, action):
        if action < 0 or action > 8:
            raise Exception(f"Action at index {action} is out of bounds")
        # Do nothing if the game has ended
        if self.state in self.get_terminal_states():
            return self.state, 0, True
        # Update state using the transition model
        if action < 8:
            action = random.choice([action, action+1])
        new_bit_value = 1 - int(self.state[action])
        next_state = self.state[0:action] + str(new_bit_value) + self.state[action+1:]
        reward = self.get_reward(self.state, action, next_state)
        self.state = next_state
        episode_end = False
        if reward != 0:
            episode_end = True
        return self.state, reward, episode_end


if __name__ == "__main__":
    
    # Check the TomAndJerry MDP
    print("\nTesting Tom and Jerry MDP")
    print("- - - - - - - - - - - - -")
    tom_and_jerry = TomAndJerry()
    s, _, _ = tom_and_jerry.reset()
    print("- Initial state: ", s)
    s, _, _ = tom_and_jerry.step(1)
    print("- New state post transition: ", s)
    p = tom_and_jerry.get_transition_prob((0,0), 2, (1,0))
    print("- Transition prob. for " + str([(0,0), 2, (1,0)]) + " is", p)
    s, a = tom_and_jerry.get_allowed_states_and_actions()
    print("- Number of states: ", len(s))
    print("- Number of actions: ", len(a))
    t_s = tom_and_jerry.get_terminal_states()
    print("- Terminal states: ", t_s)
   
    # Check the BitStrings MDP
    print("\nTesting BitStrings MDP")
    print("- - - - - - - - - - -    ")
    bit_strings = BitStrings()
    s, _, _ = bit_strings.reset()
    print("- Initial state: ", s)
    s, _, _ = bit_strings.step(1)
    print("- New state post transition: ", s)
    p = bit_strings.get_transition_prob("000000000", 2, "001000000")
    print("- Transition prob. for " + str(["000000000", 2, "001000000"]) + " is", p)
    s, a = bit_strings.get_allowed_states_and_actions()
    print("- Number of states: ", len(s))
    print("- Number of actions: ", len(a))
    t_s = bit_strings.get_terminal_states()
    print("- Terminal states: ", t_s)
    print()
