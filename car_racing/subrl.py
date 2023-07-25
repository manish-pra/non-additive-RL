# Game imports
import json
import os
import random
import time
import argparse
import errno

import numpy as np
import torch
import yaml
import wandb

import dpg_orca.VehicleModel as VehicleModel
# from icr.cpg import ARCPG, HCPG
from ORCA_training.network import Actor, Critic, GRU_Actor, GRU_Critic
from ORCA_training.orca_training_function import (
    get_reward_single, getdone, getfreezecollosionReachedreward,
    getfreezecollosionreward, getfreezereward,
    getfreezeTimecollosionReachedreward, getreward)
from support_funcs import (conjugate_gradient, critic_update, get_advantage,
                           get_relative_state, get_two_state)

workspace = "subrl/car_racing"

# parse the inputs
parser = argparse.ArgumentParser(description='A foo that bars')
parser.add_argument('-param', default="car_subrl")
parser.add_argument('-i', type=int, default=8)
args = parser.parse_args()

config = json.load(open(workspace + "/ORCA_training/config.json"))
# 1) Load the config file
with open(workspace + "/params/" + args.param + ".yaml") as file:
    params = yaml.load(file, Loader=yaml.FullLoader)

save_path =  workspace + "/experiments/13-05-23-Sat/model/" + str(args.i) + "/"

if not os.path.exists(save_path):
    try:
        os.makedirs(save_path)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise

wandb.init(
    # set the wandb project where this run will be logged
    project="car_racing_subrlf",
    mode=params["visu"]["wb"],
    config=params
)

os.environ["CUDA_VISIBLE_DEVICES"] = '0'
device = 'cpu'
vehicle_model = VehicleModel.VehicleModel(config["n_batch"], 'cpu', config)
x0 = torch.zeros(config["n_batch"], config["n_state"])
u0 = torch.zeros(config["n_batch"], config["n_control"])

state_dim = params["env"]["state_dim"]
action_dim = params["env"]["action_dim"]

if params["algo"]["recurrent"]:
    p1 = GRU_Actor(state_dim, action_dim, std=0.1).to(device)
    q = GRU_Critic(state_dim).to(device)
else:
    p1 = Actor(state_dim, action_dim, std=0.1).to(device)
    q = Critic(state_dim).to(device)
# p1 = Actor(state_dim, action_dim, std=0.1).to(device)
# p1 = GRU_Actor(state_dim, action_dim, std=0.1).to(device)
# q = Critic(state_dim).to(device)

optim_q = torch.optim.Adam(q.parameters(), lr=params["algo"]["critic_lr"])
optim_p1 = torch.optim.Adam(p1.parameters(), lr=params["algo"]["actor_lr"])

# https://github.com/seungeunrho/minimalRL/blob/master/ppo-lstm.py

num_episode = params["algo"]["num_episode"]

for t_eps in range(num_episode):
    mat_action1 = []
    mat_state1 = []
    mat_h1 = []
    mat_h2 = []
    mat_reward1 = []
    mat_done = []
    mat_hidden = []
    print(t_eps)
    curr_batch_size = params["algo"]["batch_size"]

    # data_collection
    avg_itr = 0

    state_c1 = torch.zeros(curr_batch_size, config["n_state"])
    init_p1 = torch.zeros((curr_batch_size))  # 5*torch.rand((curr_batch_size))
    state_c1[:, 0] = init_p1
    a = random.choice([-0.1, 0.1])
    state_c1[:, 1] = a*torch.ones((curr_batch_size))
    batch_mat_state1 = torch.empty(0)
    batch_mat_h1 = torch.empty(0)
    batch_mat_h2 = torch.empty(0)
    batch_mat_action1 = torch.empty(0)
    batch_mat_reward1 = torch.empty(0)
    batch_mat_done = torch.empty(0)

    itr = 0
    done = torch.tensor([False])
    done_c1 = torch.zeros((curr_batch_size)) <= -0.1
    prev_coll_c1 = torch.zeros((curr_batch_size)) <= -0.1
    counter1 = torch.zeros((curr_batch_size))
    # hidden = torch.from_numpy(np.zeros((2, 8, 128))).float()
    h_out = (torch.zeros([2, curr_batch_size, 128], dtype=torch.float), torch.zeros([2, curr_batch_size, 128], dtype=torch.float))[0]
    h_init = (torch.zeros([2, 1, 128], dtype=torch.float), torch.zeros([2, 1, 128], dtype=torch.float))[0]
    # for itr in range(50):
    while np.all(done.numpy()) == False:
        avg_itr += 1
        if params["algo"]["recurrent"]:
            h_in = h_out
            dist1, h_out = p1(state_c1[:, 0:5].reshape(1,curr_batch_size,5), h_in)
            action1 = dist1.sample().to('cpu')[0,:,:]
        else:
            dist1 = p1(state_c1[:, 0:5])
            action1 = dist1.sample().to('cpu')


        if itr > 0:
            mat_state1 = torch.cat([mat_state1.view(-1, curr_batch_size, 5), state_c1[:,
                                   0:5].view(-1, curr_batch_size, 5)], dim=0)  # concate along dim = 0
            # mat_h1 = torch.cat([mat_h1.view(-1, 2, curr_batch_size, 128), h_in[0].view(-1, 2, curr_batch_size, 128)], dim=0)
            # mat_h2 = torch.cat([mat_h2.view(-1, 2, curr_batch_size, 128), h_in[1].view(-1, 2, curr_batch_size, 128)], dim=0)
            mat_action1 = torch.cat(
                [mat_action1.view(-1, curr_batch_size, 2), action1.view(-1, curr_batch_size, 2)], dim=0)
        else:
            mat_state1 = state_c1[:, 0:5]
            # mat_h1 = h_in[0]
            # mat_h2 = h_in[1]
            mat_action1 = action1

        # mat_state2.append(state_c2[:,0:5])
        # mat_action1.append(action1)
        # mat_action2.append(action2)

        prev_state_c1 = state_c1

        state_c1 = vehicle_model.dynModelBlendBatch(
            state_c1.view(-1, 6), action1.view(-1, 2)).view(-1, 6)

        state_c1 = (state_c1.transpose(0, 1) * (~done_c1) +
                    prev_state_c1.transpose(0, 1) * (done_c1)).transpose(0, 1)

        reward1, done_c1 = get_reward_single(
            state_c1, vehicle_model.getLocalBounds(state_c1[:, 0]), 0, prev_state_c1, params["env"]["rew_type"])

        done = (done_c1) 
        mask_ele = ~done

        if itr > 0:
            mat_reward1 = torch.cat([mat_reward1.view(-1, curr_batch_size, 1),
                                    reward1.view(-1, curr_batch_size, 1)], dim=0)  # concate along dim = 0
            mat_done = torch.cat(
                [mat_done.view(-1, curr_batch_size, 1), mask_ele.view(-1, curr_batch_size, 1)], dim=0)
        else:
            mat_reward1 = reward1
            mat_done = mask_ele

        remaining_xo = ~done

        state_c1 = state_c1[remaining_xo]
        h_out = h_out[:,remaining_xo,:]#, h_out[1][:,remaining_xo,:]

        curr_batch_size = state_c1.size(0)

        if curr_batch_size < remaining_xo.size(0):
            # store data about the things that have been done
            if batch_mat_action1.nelement() == 0:
                batch_mat_state1 = mat_state1.transpose(
                    0, 1)[~remaining_xo].view(-1, 5)
                # batch_mat_h1 = mat_h1[:,:,~remaining_xo,:].view(-1, 2, 128).transpose(0,1)
                # batch_mat_h2 = mat_h2[:,:,~remaining_xo,:].view(-1, 2, 128).transpose(0,1)
                batch_mat_action1 = mat_action1.transpose(
                    0, 1)[~remaining_xo].view(-1, 2)
                batch_mat_reward1 = mat_reward1.transpose(
                    0, 1)[~remaining_xo].view(-1, 1)
                batch_mat_done = mat_done.transpose(
                    0, 1)[~remaining_xo].view(-1, 1)
                # progress_done1 = batch_mat_state1[batch_mat_state1.size(0) - 1, 0] - batch_mat_state1[0, 0]
                # progress_done2 = batch_mat_state2[batch_mat_state2.size(0) - 1, 0] - batch_mat_state2[0, 0]
                progress_done1 = torch.sum(mat_state1.transpose(0, 1)[~remaining_xo][:, mat_state1.size(
                    0)-1, 0] - mat_state1.transpose(0, 1)[~remaining_xo][:, 0, 0])
                element_deducted = ~(done_c1)
                done_c1 = done_c1[element_deducted]
            else:
                prev_size = batch_mat_state1.size(0)
                batch_mat_state1 = torch.cat([batch_mat_state1, mat_state1.transpose(0, 1)[
                                             ~remaining_xo].view(-1, 5)], dim=0)
                # for i_iter in range(mat_h1[:,:,~remaining_xo,:].shape[2]):
                #     batch_mat_h1 = torch.cat([batch_mat_h1, mat_h1[:,:,~remaining_xo,:][:,:,i_iter,:].transpose(0,1)], dim=1)
                #     batch_mat_h2 = torch.cat([batch_mat_h2, mat_h2[:,:,~remaining_xo,:][:,:,i_iter,:].transpose(0,1)], dim=1)
                # batch_mat_h2 = torch.cat([batch_mat_h2, mat_h2[:,:,~remaining_xo,:].transpose(0,2).view(2,-1,128)], dim=1)
                batch_mat_action1 = torch.cat([batch_mat_action1, mat_action1.transpose(0, 1)[
                                              ~remaining_xo].view(-1, 2)], dim=0)
                batch_mat_reward1 = torch.cat([batch_mat_reward1, mat_reward1.transpose(0, 1)[
                                              ~remaining_xo].view(-1, 1)], dim=0)
                batch_mat_done = torch.cat([batch_mat_done, mat_done.transpose(0, 1)[
                                           ~remaining_xo].view(-1, 1)], dim=0)
                progress_done1 = progress_done1 + torch.sum(mat_state1.transpose(0, 1)[~remaining_xo][:, mat_state1.size(0) - 1, 0] -
                                                            mat_state1.transpose(0, 1)[~remaining_xo][:, 0, 0])
                # progress_done1 = progress_done1 + batch_mat_state1[batch_mat_state1.size(0) - 1, 0] - batch_mat_state1[prev_size, 0]
                # progress_done2 = progress_done2 + batch_mat_state2[batch_mat_state2.size(0) - 1, 0] - batch_mat_state2[prev_size, 0]
                element_deducted = ~(done_c1)
                done_c1 = done_c1[element_deducted]

            # Update the mat_state 
            mat_state1 = mat_state1.transpose(
                0, 1)[remaining_xo].transpose(0, 1)
            # mat_h1 = mat_h1[:,:,remaining_xo,:]
            # mat_h2 = mat_h2[:,:,remaining_xo,:]
            mat_action1 = mat_action1.transpose(
                0, 1)[remaining_xo].transpose(0, 1)
            mat_reward1 = mat_reward1.transpose(
                0, 1)[remaining_xo].transpose(0, 1)
            mat_done = mat_done.transpose(0, 1)[remaining_xo].transpose(0, 1)

        # print(avg_itr,remaining_xo.size(0))

        # writer.add_scalar('Reward/agent1', reward1, t_eps)
        itr = itr + 1

        # or itr>900: #brak only if all elements in the array are true
        if np.all(done.numpy()) == True or batch_mat_state1.size(0) > 3000 or itr > 700:
            prev_size = batch_mat_state1.size(0)
            batch_mat_state1 = torch.cat(
                [batch_mat_state1, mat_state1.transpose(0, 1).reshape(-1, 5)], dim=0)
            
            batch_mat_action1 = torch.cat(
                [batch_mat_action1, mat_action1.transpose(0, 1).reshape(-1, 2)], dim=0)
            batch_mat_reward1 = torch.cat([batch_mat_reward1, mat_reward1.transpose(
                0, 1).reshape(-1, 1)], dim=0)  # should i create a false or true array?
            print("done", itr)
            print(mat_done.shape)
            # creating a true array of that shape
            mat_done[mat_done.size(
                0)-1, :, :] = torch.ones((mat_done[mat_done.size(0)-1, :, :].shape)) >= 2
            print(mat_done.shape, batch_mat_done.shape)
            if batch_mat_done.nelement() == 0:
                batch_mat_done = mat_done.transpose(0, 1).reshape(-1, 1)
                progress_done1 = 0
            else:
                batch_mat_done = torch.cat(
                    [batch_mat_done, mat_done.transpose(0, 1).reshape(-1, 1)], dim=0)
            if prev_size == batch_mat_state1.size(0):
                progress_done1 = progress_done1
            else:
                progress_done1 = progress_done1 + torch.sum(mat_state1.transpose(0, 1)[:, mat_state1.size(0) - 1, 0] -
                                                            mat_state1.transpose(0, 1)[:, 0, 0])
            print(batch_mat_done.shape)
            # print("done", itr)
            break

    # print(avg_itr)
    wandb.log({"Dist/variance_throttle_p1": dist1.variance.view(-1)[0], "Dist/variance_steer_p1": dist1.variance.view(-1)[1],
               "Reward/mean": batch_mat_reward1.mean(), "Reward/sum": batch_mat_reward1.sum(), "Progress/final_p1 ": progress_done1/params["algo"]["batch_size"], 
               "Progress/trajectory_length": itr, "Progress/agent1":batch_mat_state1[:, 0].mean()})
    
    print(batch_mat_state1.shape, itr)

    if params["algo"]["recurrent"]:
        val1 = torch.zeros(batch_mat_state1.shape[0], 1)
        h_out = h_init
        for i_roll in range(batch_mat_state1.shape[0]):
            val1[i_roll] = q(batch_mat_state1[i_roll].reshape(1,1,5), h_out) 
            if batch_mat_done[i_roll]: # if done make hidden state =1
                h_out = h_init
    else:
        val1 = q(batch_mat_state1)
    val_detach = val1.clone().detach().to('cpu')
    next_value = 0  # because currently we end ony when its done which is equivalent to no next state
    returns_np1 = get_advantage(
        next_value, batch_mat_reward1, val_detach, batch_mat_done, gamma=0.99, tau=0.95)

    returns1 = torch.cat(returns_np1)
    advantage_mat1 = returns1.view(1, -1) - val_detach.transpose(0, 1)

    returns1_gpu = returns1.view(-1, 1).to(device)

    # val_detach= q(state_mb)
    critic_loss = (returns1_gpu - val1).pow(2).mean()

    optim_q.zero_grad()
    critic_loss.backward()
    optim_q.step()
    # del val_loc
    loss_critic = critic_loss.detach().cpu().numpy()
    del critic_loss
    wandb.log({"Loss/critic": loss_critic})
    ed_q_time = time.time()
    # for loss_critic, gradient_norm in critic_update(batch_mat_state1, returns1_gpu, q, optim_q):
        
    #     # print('critic_update')
    # ed_q_time = time.time()
    # # print('q_time',ed_q_time-st_q_time)

    # val1_p = -advantage_mat1#val1.detach()
    val1_p = advantage_mat1.to(device)
    
    # st_time = time.time()
    # calculate gradients
    # batch_mat_action1_gpu = batch_mat_action1.to(device)
    if params["algo"]["recurrent"]:
        log_probs1 = torch.zeros(batch_mat_state1.shape[0], 1)
        h_out = h_init
        for i_roll in range(batch_mat_state1.shape[0]):
            dist_batch1, h_out = p1(batch_mat_state1[i_roll].reshape(1,1,5), h_out) 
            log_probs1_inid = dist_batch1.log_prob(batch_mat_action1[i_roll])
            log_probs1[i_roll] = log_probs1_inid.sum(-1)
            if batch_mat_done[i_roll]: # if done make hidden state =1
                h_out = h_init
    else:
        dist_batch1 = p1(batch_mat_state1)
        # dist_batch1, hidden = p1(batch_mat_state1, hidden)
        log_probs1_inid = dist_batch1.log_prob(batch_mat_action1)
        log_probs1 = log_probs1_inid.sum(1)

    optim_p1.zero_grad()
    loss = -(log_probs1*advantage_mat1).mean() - \
        params["algo"]["ent_coeff"]*dist_batch1.entropy().mean()
    loss.backward()
    optim_p1.step()
    # improve1, improve2, lamda, lam1, lam2, esp, stat = optim.step(
    #     advantage_mat1, batch_mat_state1, state_gpu_p2, batch_mat_action1, batch_mat_action2)
    ed_time = time.time()

    wandb.log({"Advantage/agent1": advantage_mat1.mean(), 'Entropy/agent1': dist_batch1.entropy().mean().detach()})
    torch.save(p1.state_dict(), save_path + '/agent1_' + str(t_eps) + ".pth")
    torch.save(q.state_dict(), save_path + '/val_' + str(t_eps) + ".pth")

