import os
import torch as T
import torch.nn as nn
import torch.optim as optim

class CriticNetwork(nn.Module):
    def __init__(self, input_dims, alpha, fc1_dims=64, fc2_dims=64,
            chkpt_dir='tmp'):
        super(CriticNetwork, self).__init__()

        self.checkpoint_file = os.path.join(chkpt_dir, 'critic_')
        input = 1
        for i in input_dims:
            input *= i
        self.critic = nn.Sequential(
                nn.Linear(input, fc1_dims),
                nn.ReLU(),
                nn.Linear(fc1_dims, fc2_dims),
                nn.ReLU(),
                nn.Linear(fc2_dims, 1)
        )

        self.optimizer = optim.Adam(self.parameters(), lr=alpha)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, state):
        # print()
        value = self.critic(state.view(state.size(0), -1))

        return value

    def save_checkpoint(self, name):
        file = os.path.join(self.checkpoint_file + name)
        T.save(self.state_dict(), file)

    def load_checkpoint(self, name):
        file = os.path.join(self.checkpoint_file + name)
        self.load_state_dict(T.load(file))  