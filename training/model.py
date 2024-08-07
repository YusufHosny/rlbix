import torch
from torch import nn

d_e = 100
d_f = 50

class DQN(nn.Module):

    def __init__(self, n_observations, n_actions):
        super(DQN, self).__init__()
        
        self.embedding = nn.Embedding(n_observations, d_e)
        self.cnn = nn.Sequential(
            nn.Conv3d(d_e, d_f, 2, padding='valid'),
            nn.ELU(),
            nn.Conv3d(d_f, d_f, 2, padding='same'),
            nn.ELU(),
            nn.Conv3d(d_f, d_f, 2, padding='same'),
            nn.ELU()
        )
        self.fc = nn.Sequential(
            nn.Linear(8*d_f, 50),
            nn.ELU(),
            nn.Linear(50, 50),
            nn.ELU(),
            nn.Linear(50, 50),
            nn.ELU()
        )
        self.output = nn.Linear(50, n_actions)

    def forward(self, x):
        # [B, 3, 3, 3] embed-> [B, 3, 3, 3, 50] permute-> [B, 50, 3, 3, 3] conv3d-> [B, 20, 2, 2, 2] flatten-> [B, 160] fc-> [B, 50] out-> [B, 12] -> out
        x = self.embedding(x)
        x = x.permute(0, 4, 1, 2, 3)
        x = self.cnn(x)
        x = x.reshape(-1, 8*d_f)
        x = self.fc(x)
        x = self.output(x)
        return x
