import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.normal_(m.weight, mean=0., std=0.1)
        nn.init.constant_(m.bias, 0.1)

class ActorVar(nn.Module):
    def __init__(self, state_dim, action_dim, std=1.0):
        super(ActorVar, self).__init__()

        self.actor = nn.Sequential(nn.Linear(state_dim, 128),  # 84*50
                                   nn.Tanh(),
                                   nn.Linear(128, 128),  # 50*20
                                   nn.Tanh())

        self.actor_tail = nn.Sequential(nn.Linear(128, action_dim),
                                   nn.Tanh())  # 20*2

        self.var_tail = nn.Sequential(nn.Linear(128, action_dim))

        # self.log_std = nn.Parameter(torch.ones(action_dim) * std)

        self.apply(init_weights)

    def forward(self, state):
        x = self.actor(state)
        mu = self.actor_tail(x)
        log_std = self.var_tail(x)
        log_std_clamped = torch.clamp(log_std, min=-20, max=0.1)
        std = log_std_clamped.exp().expand_as(mu)
        # std = self.log_std.exp().expand_as(mu)
        dist = Normal(mu,std)
        return dist

class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, std=1.0):
        super(Actor, self).__init__()

        self.actor = nn.Sequential(nn.Linear(state_dim, 128),  # 84*50
                                   nn.ReLU(),
                                   nn.Linear(128, 128),  # 50*20
                                   nn.ReLU(),
                                   nn.Linear(128, action_dim),
                                   nn.Tanh())  # 20*2

        self.log_std = nn.Parameter(torch.ones(action_dim) * std)

        self.apply(init_weights)

    def forward(self, state):
        mu = self.actor(state)
        log_std_clamped = self.log_std#torch.clamp(self.log_std, min=-20, max=0.1)
        std = log_std_clamped.exp().expand_as(mu)
        dist = Normal(mu,std)
        return dist

class Critic(nn.Module):
    def __init__(self, state_dim):
        super(Critic, self).__init__()

        self.critic = nn.Sequential(nn.Linear(state_dim, 128),  # 84*50
                                    nn.ReLU(),
                                    nn.Linear(128, 128),  # 50*20
                                    nn.ReLU(),
                                    nn.Linear(128, 1))  # 20*2

        self.apply(init_weights)

    def forward(self, state):
        value = self.critic(state)
        return value
    
class RNNActor():
    def __init__(self, state_dim, action_dim, std=1.0):
        super(Actor, self).__init__()

        self.actor = nn.Sequential(nn.Linear(state_dim, 128),  # 84*50
                                   nn.ReLU(),
                                   nn.Linear(128, 128),  # 50*20
                                   nn.ReLU(),
                                   nn.Linear(128, action_dim),
                                   nn.Tanh())  # 20*2

        self.log_std = nn.Parameter(torch.ones(action_dim) * std)

        self.apply(init_weights)

    def forward(self, state):
        mu = self.actor(state)
        log_std_clamped = self.log_std#torch.clamp(self.log_std, min=-20, max=0.1)
        std = log_std_clamped.exp().expand_as(mu)
        dist = Normal(mu,std)
        return dist
    rnn = nn.RNN(input_size = 20, 
             hidden_size = 10, 
             num_layers = 4,)   
    
# class GRUActor(nn.Module):
#     def __init__(self, state_dim, action_dim, std=1.0):
#         super(GRUActor, self).__init__()

#         self.rnn = nn.GRU(input_size=state_dim, hidden_size=128, num_layers=2, batch_first=True)
#         self.fc_mu = nn.Linear(128, action_dim)
#         self.fc_log_std = nn.Linear(128, action_dim)

#         self.log_std = nn.Parameter(torch.ones(action_dim) * std)
#         self.apply(init_weights)

#     def forward(self, state):
#         _, hn = self.rnn(state)
#         hn = hn[-1]
#         mu = self.fc_mu(hn)
#         log_std_clamped = self.log_std#torch.clamp(self.log_std, min=-20, max=0.1)
#         std = log_std_clamped.exp()
#         dist = Normal(mu, std)
#         return dist
    
class GRU_Actor(nn.Module):
    def __init__(
        self, state_dim, action_dim, std=1.0):
        super(GRU_Actor, self).__init__()
        self.recurrent = True
        self.action_dim = action_dim

        if self.recurrent:
            self.l1 = nn.GRU(input_size=state_dim, hidden_size=128, num_layers=2, batch_first=False)
            self.actor = nn.Sequential(nn.Linear(128, action_dim),
                                       nn.Tanh())
        else:
            self.actor = nn.Sequential(nn.Linear(state_dim, 128),  # 84*50
                            nn.ReLU(),
                            nn.Linear(128, 128),  # 50*20
                            nn.ReLU(),
                            nn.Linear(128, action_dim),
                            nn.Tanh())  # 20*2

        self.log_std = nn.Parameter(torch.ones(action_dim) * std)
        self.apply(init_weights)

    def forward(self, state, hidden):
        if self.recurrent:
            # self.l1.flatten_parameters()
            p, h = self.l1(state, hidden)
            mu = self.actor(p.data)
        else:
            mu, h = self.actor(state), None

        # mu = torch.tanh(self.l2(p.data))
        log_std_clamped = self.log_std#torch.clamp(self.log_std, min=-20, max=0.1)
        std = log_std_clamped.exp().expand_as(mu)
        dist = Normal(mu,std)
        return dist, h
    
class GRU_Critic(nn.Module):
    def __init__(self, state_dim):
        super(GRU_Critic, self).__init__()
        self.l1 = nn.GRU(input_size=state_dim, hidden_size=128, num_layers=2, batch_first=False)
        self.critic = nn.Linear(128, 1)

        self.apply(init_weights)

    def forward(self, state, hidden):
        p, h = self.l1(state, hidden)
        value = self.critic(p.data)
        return value