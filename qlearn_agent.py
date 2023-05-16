import random
import numpy as np


class Q_learn:
    def __init__(self, state):
        self.init_state_index = self.state_index_map(state)
        self.state_index = self.init_state_index
        self.next_state_index = None
        self.action_index = 1
        # Initialize Q-table
        self.num_states = (58,  # num of bird_y buckets
                           2,   # num of bird_velocity buckets
                           16,  # num of gap_x buckets
                           24   # num of gap_y buckets
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
        self.action_index = 1

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

        state_index = ()  # It's a tuple to be used in indexing a np array
        return state_index

    # Define a function to select an action using epsilon-greedy strategy
    def epsilon_greedy(self, state_index):
        # Choose a random action with probability epsilon
        if random.uniform(0, 1) < self.epsilon:
            action_index = random.randrange(0, self.Q[state_index].size)
        # Otherwise, choose the action with the highest Q-value
        else:
            max_value = max(self.Q[state_index])
            actions_indices = [i for i, v in self.Q[state_index] if v == max_value]
            action_index = random.choice(actions_indices)

        self.action_index = action_index
        return action_index

    def take_action(self, state, exploration=True):
        """
        It maps "action_index = 1" to "jump"
        and "action_index = 0" to " " meaning no jump
        """
        state_index = self.state_index_map(state)
        if exploration:  # True during learning
            action = "jump" if self.epsilon_greedy(state_index) else " "
        else:
            max_value = max(self.Q[state_index])
            actions_indices = [i for i, v in self.Q[state_index] if v == max_value]
            action_index = random.choice(actions_indices)
            action = "jump" if action_index else " "
        return action

    # Q-learning algorithm
    def learn(self, state, reward, done=False):
        self.next_state_index = self.state_index_map(state)
        # Update Q-value for state-action pair
        td_error = reward + self.gamma * np.max(self.Q[self.next_state_index]) - self.Q[self.state_index + (self.action_index,)]
        self.Q[self.state_index + (self.action_index,)] += self.alpha * td_error

        # update state
        self.state_index = self.next_state_index

        # when episode ends reset the agent
        if done:
            self.reset()
            # print number of complete episodes
            print(self.num_episodes)
            self.num_episodes += 1
