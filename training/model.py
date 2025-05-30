import torch
from torch import nn

d_c = 6 # n of channels
d_fc1 = 100 # fc1 layers dimensions
d_f = 50 # filter count for conv layers
d_fco = 100 # fc2 layers dimensions
k_d = 2 # conv kernel dim
d_co = 2 # output dimension of conv layers (d_co * d_co * d_co)
d_co3 = d_co ** 3 # dim after flattening

class DQN(nn.Module):

    def __init__(self, n_actions):
        super(DQN, self).__init__()
        
        self.fc1 = nn.Sequential(
            nn.Linear(d_c, d_fc1),
            nn.ELU(),
            nn.Linear(d_fc1, d_fc1),
            nn.ELU(),
        )

        self.cnn = nn.Sequential(
            nn.Conv3d(d_fc1, d_f, k_d, padding='valid'),
            nn.ELU(),
            nn.Conv3d(d_f, d_f, k_d, padding='valid'),
            nn.ELU(),
            nn.Conv3d(d_f, d_f, k_d, padding='valid'),
            nn.ELU(),
        )
        self.fc2 = nn.Sequential(
            nn.Linear(d_co3*d_f, d_fco),
            nn.ELU(),
            nn.Linear(d_fco, d_fco),
            nn.ELU(),
            nn.Linear(d_fco, d_fco),
            nn.ELU(),
        )
        self.output = nn.Linear(d_fco, n_actions)

    def forward(self, x):
        # [B, 6, 5, 5, 5] permute-> [B, 5, 5, 5, 6] fc1-> [B, 5, 5, 5, d_e] permute-> [B, d_e, 5, 5, 5]
        # conv3d-> [B, d_f, d_co, d_co, d_co] flatten-> [B, d_co3 *d_f] fc2-> [B, d_fco] out-> [B, 12] -> out
        x = torch.permute(x, (0, 2, 3, 4, 1))
        x = self.fc1(x)
        x = torch.permute(x, (0, 4, 1, 2, 3))
        x = self.cnn(x)
        x = x.reshape(-1, d_co3 *d_f)
        x = self.fc2(x)
        x = self.output(x)
        return x
