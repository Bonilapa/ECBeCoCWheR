import tensorflow as tf
import keras as ks
from keras.layers import Dense

class ActorNetwork(ks.Model):
    def __init__(self, n_actions, fc1_dims=256, fc2_dims=256):
        super(ActorNetwork, self).__init__()

        self.fc_common_1 = Dense(fc1_dims, activation='relu')
        self.fc_common_2 = Dense(fc2_dims, activation='relu')

        self.task_fcs = []
        self.task_outs = []
        for i in range(n_actions):
            self.task_fcs.append(Dense(15, activation='relu'))
            self.task_outs.append(Dense(1, activation='linear'))

    def call(self, state):
        
        x = self.fc1(state)
        x = self.fc2(x)
        x = self.fc3(x)
        return x

class CriticNetwork(ks.Model):
    def __init__(self, fc1_dims=256, fc2_dims=256):
        super(CriticNetwork, self).__init__()

        self.fc1 = Dense(fc1_dims, activation='relu')
        self.fc2 = Dense(fc2_dims, activation='relu')
        self.fc3 = Dense(1, activation=None)
        
    def call(self, state):
        x = self.fc1(state)
        x = self.fc2(x)
        x = self.fc3(x)

        return x
