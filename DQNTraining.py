#import libraries as needed
from NeuralNetworkClass import Network
from DenseLayerClass import Dense
from collections import deque
import numpy as np

class DQN():
    def __init__(self, observation_size, action_size, discount_factor, n_episodes, start_epsilon, end_epsilon, epsilon_decay, batch_size):
        #Observation and action space size
        self.observation_size = observation_size
        self.action_size = action_size

        #Q Learning members
        self.discount_factor = discount_factor
        self.n_episodes = n_episodes
        self.epsilon = start_epsilon
        self.end_epsilon = end_epsilon
        self.epsilon_decay = epsilon_decay
        
        #replay memory members
        self.memory = deque(maxlen = 1000)
        self.batch_size = batch_size
        #these are used for making weight updates
        self.update_num = 50
        self.update_steps = 0

        #neural network members
        self.q_network = self.create_model()
        self.target_network = self.create_model()

    #will be a 4 layer NN, using MSE
    def create_model(self):
        return model

    #add to our experience replay
    def add_memory(self, curr_state, action, next_state, reward, truncated, terminated):
        pass
    
    #epsilon greedy approach for an action
    def get_action(self, curr_state):
        pass

    def choose_action(self, curr_state):
        pass
    
    #function to decay epsilon, call it at the end of train
    def decay_epsilon(self):
        self.epsilon = max(self.end_epsilon, self.epsilon - self.epsilon_decay)

    #function to train DQN
    def train(self):
        pass

    #function to update our target network with the weights of the q network
    def update_target(self):
        pass

    #methods to save model
    def save_model(self, file_name):
        pass
    
    #method to load model
    def load_model(self, file_name):
        pass