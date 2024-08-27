import torch
from torch import nn

d_c = 6
d_e = 100
d_f = 50
d_o = 50

k_d = 3
k_d3 = k_d**3

class DQN(nn.Module):

    def __init__(self, n_actions):
        super(DQN, self).__init__()
        
        self.fc1 = nn.Sequential(
            nn.Linear(d_c, d_e),
            nn.ELU(),
            nn.Linear(d_e, d_e),
            nn.ELU(),
        )

        self.cnn = nn.Sequential(
            nn.Conv3d(d_e, d_f, k_d, padding='valid'),
            nn.ELU(),
        )
        self.fc2 = nn.Sequential(
            nn.Linear(k_d3*d_f, d_o),
            nn.ELU(),
            nn.Linear(d_o, d_o),
            nn.ELU(),
        )
        self.output = nn.Linear(d_o, n_actions)

    def forward(self, x):
        # [B, 6, 5, 5, 5] permute-> [B, 5, 5, 5, 6] fc1-> [B, 5, 5, 5, d_e] permute-> [B, d_e, 5, 5, 5]
        # conv3d-> [B, d_f, k_d, k_d, k_d] flatten-> [B, k_d^3 *d_f] fc2-> [B, d_o] out-> [B, 12] -> out
        x = torch.permute(x, (0, 2, 3, 4, 1))
        x = self.fc1(x)
        x = torch.permute(x, (0, 4, 1, 2, 3))
        x = self.cnn(x)
        x = x.reshape(-1, k_d3 *d_f)
        x = self.fc2(x)
        x = self.output(x)
        return x
