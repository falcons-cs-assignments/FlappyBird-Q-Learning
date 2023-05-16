import random
import numpy as np


class Q_learn:
    def __init__(self, state):
        self.init_state_index = self.state_index_map(state)
        self.state_index = self.init_state_index
        self.next_state_index = None
        self.action = "jump"
        # Initialize Q-table
        self.num_states = (58,  # buckets num of bird_y
                           2,   # buckets num of bird_velocity
                           16,  # buckets num of gap_x
                           24   # buckets num of gap_y
                           )
        self.num_actions = (2,)  # It's a tuple
        self.Q = np.zeros(self.num_states + self.num_actions)

        # Set hyper parameters
        self.alpha = 0.1
        self.gamma = 0.01
        self.num_episodes = 0

        # Define epsilon (the exploration rate)
        self.epsilon = 0.1

    def reset(self):
        self.state_index = self.init_state_index
        self.action = "jump"

    def state_index_map(self, state):
        # Todo: convert continuous state to a discrete index
        # Todo: Take into your account that ... There is a special state at the start of the game "pipe_x variable is at the very right"
        """
        # input form:
        state = {
            'bird_y':
            'bird_v':
            'pipe_positions':
            'score':
            'game_state':
        }

        # ranges of important variables of the state:
        state = {
            'bird_y': [144:720]     bucket_size: 10
            'bird_v': [-15:+5]      index: '0' or '1' for 'up' or 'down'
            'pipe_positions': (
                                pipe.gap_x: [93:393]        bucket_size: 20
                                pipe.gap_y: [314:550]       bucket_size: 10
                              )
        }
        """

        state_index = {

        }
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
        self.next_state_index = self.state_index_map(state)
        # Update Q-value for state-action pair
        td_error = reward + self.gamma * np.max(self.Q[self.next_state_index]) - self.Q[self.state_index][self.action]
        self.Q[self.state_index][self.action] += self.alpha * td_error

        # update state
        self.state_index = self.next_state_index

        # when episode ends reset the agent
        if done:
            self.reset()
            # print number of complete episodes
            print(self.num_episodes)
            self.num_episodes += 1
