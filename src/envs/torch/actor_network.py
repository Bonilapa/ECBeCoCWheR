import os
import torch as T
import torch.nn as nn
import torch.optim as optim
from torch.distributions.categorical import Categorical

class ActorNetwork(nn.Module):
    def __init__(self, n_actions, input_dims, alpha,
            fc1_dims=10, fc2_dims=10, chkpt_dir='tmp'):
        super(ActorNetwork, self).__init__()
        # print(n_actions, *input_dims)
        self.checkpoint_file = os.path.join(chkpt_dir, 'actor_')
        input = 1
        for i in input_dims:
            input *= i
        self.actor = nn.Sequential(
                nn.Linear(input, fc1_dims),
                nn.ReLU(),
                nn.Linear(fc1_dims, fc2_dims),
                nn.ReLU(),
                nn.Linear(fc2_dims, 3)
        )

        self.optimizer = optim.Adam(self.parameters(), lr=alpha)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, state):
        # print(state.shape, "forward")
        dist = self.actor(state.view(state.size(0), -1))
        # print(dist)
        # dist = Categorical(dist)
        
        return dist

    def save_checkpoint(self, name):
        file = os.path.join(self.checkpoint_file + name)
        T.save(self.state_dict(), file)

    def load_checkpoint(self, name):
        file = os.path.join(self.checkpoint_file + name)
        self.load_state_dict(T.load(file))