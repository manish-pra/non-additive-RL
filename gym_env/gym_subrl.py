# Game imports
import gc
import json
import os
import random
import sys
import time
import wandb

import numpy as np
import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical, Normal
from torch.utils.tensorboard import SummaryWriter

import gymnasium as gym
import numpy as np
from network import Actor, Critic, GRU_Actor, GRU_Critic
from helper import get_advantage, critic_update
import argparse
import yaml
from inheritance_test import MyAntEnv

workspace = "subrl/gym_env"

parser = argparse.ArgumentParser(description="A foo that bars")
parser.add_argument("-param", default="gym_subRL")  # params

parser.add_argument("-env", type=int, default=0)
parser.add_argument("-i", type=int, default=8)  # initialized at origin
args = parser.parse_args()

# 1) Load the config file
with open(workspace + "/params/" + args.param + ".yaml") as file:
    params = yaml.load(file, Loader=yaml.FullLoader)
print(params)


env_load_path = (
    workspace + "/experiments/12-01-23/environments/env_" + str(args.env) + "/")

save_path = env_load_path + "/" + args.param + "/"

if not os.path.exists(save_path):
    try:
        os.makedirs(save_path)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise

wandb.init(
    # set the wandb project where this run will be logged
    project="gym-PG-prac",
    mode=params["visu"]["wb"],
    config=params,
)

# envs = gym.vector.AsyncVectorEnv([lambda: gym.make("Ant-v4") for i in range(50)])#.make("Ant-v4", num_envs=50)#, render_mode="human")
# envs = gym.make("Ant-v4", exclude_current_positions_from_observation=False)
envs = gym.vector.AsyncVectorEnv([lambda: MyAntEnv(rew_type=params["env"]["rew_type"],exclude_current_positions_from_observation=False)]*params["env"]["num_envs"])
# envs = gym.vector.make(params["env"]["name"], num_envs=params["env"]["num_envs"], exclude_current_positions_from_observation=False)
envs.reset()
obs_dim = envs.observation_space.shape[1]
act_dim = envs.action_space.shape[1]
# actions = np.array([1, 0, 1,1, 0, 1,1, 0, 1, 0, 1,1, 0, 1,1, 0, 1, 0, 1,1, 0, 1,1, 0]).reshape(3,8)
# observations, rewards, dones, infos, _ = envs.step(actions)
# envs.render()

device = "cpu" #torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
if params["algo"]["recurrent"]:
    p1 = GRU_Actor(obs_dim, act_dim, std=0.1).to(device)
    q = GRU_Critic(obs_dim).to(device)
else:
    p1 = Actor(obs_dim, act_dim, std=0.1).to(device)
    q = Critic(obs_dim).to(device)

# p1.load_state_dict(
#     torch.load(workspace + '/experiments/23-01-23/' + 'model/agent1_3000.pth'))
# q.load_state_dict(
#     torch.load(workspace + '/experiments/23-01-23/' + 'model/val_3000.pth'))

optim_q = torch.optim.Adam(q.parameters(), lr=params["algo"]["actor_lr"])
optim_p1 = torch.optim.Adam(p1.parameters(), lr=params["algo"]["critic_lr"])

num_episode = params["algo"]["num_episode"]
traj_len = 400
curr_batch_size = params["env"]["num_envs"]
b_lim  = params["env"]["b_lim"]
reward = torch.empty(params["env"]["num_envs"], 1).to(device)
rew_traj = torch.empty(params["env"]["num_envs"], 1).to(device)
for t_eps in range(num_episode):

    batch_mat_state = torch.empty(traj_len,params["env"]["num_envs"], obs_dim).to(device)
    batch_h1 = torch.empty(traj_len,2,params["env"]["num_envs"], 128).to(device)
    batch_h2 = torch.empty(traj_len,2,params["env"]["num_envs"], 128).to(device)
    batch_mat_action = torch.empty(traj_len,params["env"]["num_envs"], act_dim).to(device)
    batch_mat_reward = torch.empty(traj_len,params["env"]["num_envs"], 1).to(device)
    batch_mat_coverage = torch.empty(traj_len,params["env"]["num_envs"], 1).to(device)
    batch_mat_done = torch.empty(traj_len,params["env"]["num_envs"], 1).to(device)
    mask = torch.zeros(params["env"]["num_envs"], 100,100)
    print(t_eps)

    state, info = envs.reset()
    state = torch.from_numpy(state).float().to(device)

    h_out = (torch.zeros([2, params["env"]["num_envs"], 128], dtype=torch.float), torch.zeros([2, params["env"]["num_envs"], 128], dtype=torch.float))[0]
    h_init = h_out
    for itr in range(traj_len):
        # print(itr)

        if params["algo"]["recurrent"]:
            h_in = h_out
            dist, h_out = p1(state.reshape(1,params["env"]["num_envs"],29), h_in)
            action = dist.sample().to(device)[0,:,:]
            # batch_h1[itr,:,:,:], batch_h2[itr,:,:,:] = h_in
            batch_h1[itr,:,:,:] = h_in
        else:
            dist = p1(state)
            action = dist.sample().to(device)

        batch_mat_state[itr,:,:] = state
        batch_mat_action[itr,:,:] = action
        
        state, reward_not, terminated, truncated, info = envs.step(action.cpu().numpy())

        state = torch.from_numpy(state).float().to(device)
        reward_not = torch.from_numpy(reward_not).float().to(device)
        coverage = torch.from_numpy(info['coverage']).float().to(device)
        done = torch.from_numpy(terminated).float().to(device)
        if terminated.any():
            h_out[:,terminated,:] = torch.zeros_like(h_out[:,terminated,:])

        batch_mat_coverage[itr,:,:] = coverage.view(-1,1)
        batch_mat_reward[itr,:,:] = reward_not.view(-1,1)
        batch_mat_done[itr,:,:] = done.view(-1,1)*1

    batch_mat_done[itr,:,:] = torch.ones_like(done.view(-1,1))

    wandb.log({"reward": batch_mat_reward.mean(), "coverage_mean": batch_mat_coverage.mean(), "coverage_final": batch_mat_coverage[-1,:,:].mean()})
    if params["algo"]["recurrent"]:
        val1 = torch.zeros(batch_mat_state.shape[0], params["env"]["num_envs"],1)
        for i_roll in range(batch_mat_state.shape[0]):
            # dist_batch1, _ = p1(batch_mat_state[i_roll].reshape(1,params["env"]["num_envs"],29), )
            # val1[i_roll] = q(batch_mat_state[i_roll].reshape(1,params["env"]["num_envs"],29), (batch_h1[i_roll],batch_h2[i_roll]))
            val1[i_roll] = q(batch_mat_state[i_roll].reshape(1,params["env"]["num_envs"],29), batch_h1[i_roll])
    else:
        val1 = q(batch_mat_state)
    next_value = 0  # because currently we end ony when its done which is equivalent to no next state
    returns_np1 = get_advantage(
        next_value, batch_mat_reward, val1.detach(), batch_mat_done, gamma=0.99, tau=0.95, device=device
    )

    critic_loss = (returns_np1 - val1).pow(2).mean()

    optim_q.zero_grad()
    critic_loss.backward()
    optim_q.step()
    wandb.log({"critic_loss": critic_loss.item()})
    
    advantage_mat1 = (returns_np1 - val1).detach()[:,:,0]
    del returns_np1

    wandb.log({"advantage": advantage_mat1.mean()})
    # calculate gradients
    if params["algo"]["recurrent"]:
        log_probs1 = torch.zeros(batch_mat_state.shape[0], params["env"]["num_envs"])
        for i_roll in range(batch_mat_state.shape[0]):
            # dist_batch1, _ = p1(batch_mat_state[i_roll].reshape(1,params["env"]["num_envs"],29), (batch_h1[i_roll],batch_h2[i_roll]))
            dist_batch1, _ = p1(batch_mat_state[i_roll].reshape(1,params["env"]["num_envs"],29), batch_h1[i_roll])
            log_probs1_inid = dist_batch1.log_prob(batch_mat_action[i_roll])
            log_probs1[i_roll] = log_probs1_inid.sum(-1)
    else:
        dist_batch1 = p1(batch_mat_state)
        log_probs1_inid = dist_batch1.log_prob(batch_mat_action)
        log_probs1 = log_probs1_inid.sum(-1)

    optim_p1.zero_grad()
    loss = -(log_probs1 * advantage_mat1).mean() - 0.01 * dist_batch1.entropy().mean()
    loss.backward()
    optim_p1.step()

    if t_eps % 100 == 0:
            torch.save(p1.state_dict(), save_path + 'agent1_' + str(t_eps) + ".pth")
            torch.save(q.state_dict(), save_path + 'val_' + str(t_eps) + ".pth")