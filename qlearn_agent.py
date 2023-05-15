import random
import numpy as np


class Q_learn:
    def __init__(self, state):
        self.init_state = self.state_index_map(state)
        self.state = self.init_state
        self.next_state = None
        self.action = "jump"
        # Initialize Q-table
        self.num_states = ...
        self.num_actions = 2
        self.Q = np.zeros((self.num_states, self.num_actions))

        # Set hyper parameters
        self.alpha = 0.1
        self.gamma = 0.01
        self.num_episodes = 0

        # Define epsilon (the exploration rate)
        self.epsilon = 0.1

    def reset(self):
        self.state = self.init_state
        self.action = "jump"

    def state_index_map(self, state):
        index = ...
        return index
        pass

    # Define a function to select an action using epsilon-greedy strategy
    def epsilon_greedy(self, state):
        # Choose a random action with probability epsilon
        if random.uniform(0, 1) < self.epsilon:
            action = random.choice(list(self.Q[state].keys()))
        # Otherwise, choose the action with the highest Q-value
        else:
            max_value = max(self.Q[state].values())
            actions = [a for a, v in self.Q[state].items() if v == max_value]
            action = random.choice(actions)
        return action

    def take_action(self, state, exploration=True):
        if exploration:  # True during learning
            self.action = self.epsilon_greedy(state)
        else:
            max_value = max(self.Q[state].values())
            actions = [a for a, v in self.Q[state].items() if v == max_value]
            self.action = random.choice(actions)
        return self.action

    # Q-learning algorithm
    def learn(self, state, reward, done=False):
        self.next_state = state
        # Update Q-value for state-action pair
        td_error = reward + self.gamma * np.max(self.Q[self.next_state]) - self.Q[self.state][self.action]
        self.Q[self.state][self.action] += self.alpha * td_error

        # update state
        self.state = self.next_state

        # when episode ends reset the agent
        if done:
            self.reset()
            # print number of complete episodes
            print(self.num_episodes)
            self.num_episodes += 1
