
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal


class policy(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(policy, self).__init__()
        # self.actor = nn.Sequential(nn.Linear(state_dim, action_dim),  # 2(self and other's position) * 2(action)
        #                            nn.Softmax(dim =-1))  # 20*2
        self.actor = nn.Sequential(nn.Linear(state_dim, 64),  # 84*50
                                   nn.Tanh(),
                                   nn.Linear(64, 32),  # 50*20
                                   nn.Tanh(),
                                   nn.Linear(32, action_dim),
                                   nn.Softmax(dim=-1))

    def forward(self, state):
        mu = self.actor(state)
        return mu


def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.normal_(m.weight, mean=0., std=0.1)
        nn.init.constant_(m.bias, 0.1)

def append_state(mat_state, H):
    cur_H = len(mat_state)
    mat_state = torch.vstack(mat_state)
    dummy = -1*torch.ones((H-cur_H, mat_state.shape[1]))
    batch_state = torch.cat([mat_state, dummy]).transpose(0, 1)
    return batch_state