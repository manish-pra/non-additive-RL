import argparse
import errno
import os
import random
from importlib.metadata import requires
from timeit import timeit
import dill as pickle
import numpy as np
import scipy
import torch
import wandb
import yaml
from sympy import Matrix, MatrixSymbol, derive_by_array, symarray
from torch.distributions import Categorical

from utils.environment import GridWorld
from utils.network import append_state
from utils.network import policy as agent_net
from utils.visualization import Visu

# TODO: 1. remove dependence from matrix and could run multiple times in parallel, .sh script, run it on the server, check how to plot multiple on wb,
# can it plot different kappa's on the same with grouping
# apply a policy gradient algorithm, since the policy is deterministic, use policy iteration/value iteration since we know the dynamics
#
workspace = "subrl"

parser = argparse.ArgumentParser(description='A foo that bars')
parser.add_argument('-param', default="subrl")  # params

parser.add_argument('-env', type=int, default=1)
parser.add_argument('-i', type=int, default=8)  # initialized at origin
args = parser.parse_args()

# 1) Load the config file
with open(workspace + "/params/" + args.param + ".yaml") as file:
    params = yaml.load(file, Loader=yaml.FullLoader)
print(params)

# 2) Set the path and copy params from file
env_load_path = workspace + \
    "/environments/" + params["env"]["node_weight"]+ "/env_" + \
    str(args.env)

params['env']['num'] = args.env 
# start a new wandb run to track this script
wandb.init(
    # set the wandb project where this run will be logged
    project="code-" + params["env"]["node_weight"],
    mode=params["visu"]["wb"],
    config=params
)

epochs = params["alg"]["epochs"]

H = params["env"]["horizon"]
MAX_Ret = 2*(H+1)
if params["env"]["disc_size"] == "large":
    MAX_Ret = 3*(H+2)

# 3) Setup the environement
env = GridWorld(
    env_params=params["env"], common_params=params["common"], visu_params=params["visu"], env_file_path=env_load_path)
node_size = params["env"]["shape"]['x']*params["env"]["shape"]['y']
# TransitionMatrix = torch.zeros(node_size, node_size)

if params["env"]["node_weight"] == "entropy" or params["env"]["node_weight"] == "steiner_covering" or params["env"]["node_weight"] == "GP": 
    a_file = open(env_load_path +".pkl", "rb")
    data = pickle.load(a_file)
    a_file.close()

if params["env"]["node_weight"] == "entropy":
    env.cov = data
if params["env"]["node_weight"] == "steiner_covering":
    env.items_loc = data
if params["env"]["node_weight"] == "GP":
    env.weight = data

visu = Visu(env_params=params["env"])
# plt, fig = visu.stiener_grid( items_loc=env.items_loc, init=34)
# wandb.log({"chart": wandb.Image(fig)})
# plt.close()
# Hori_TransitionMatrix = torch.zeros(node_size*H, node_size*H)
# for node in env.horizon_transition_graph.nodes:
#     connected_edges = env.horizon_transition_graph.edges(node)
#     for u, v in connected_edges:
#         Hori_TransitionMatrix[u[0]*node_size+u[1], v[0]*node_size + v[1]] = 1.0
env.get_horizon_transition_matrix()
# policy = Policy(TransitionMatrix=TransitionMatrix, Hori_TransitionMatrix=Hori_TransitionMatrix, ActionTransitionMatrix=env.Hori_ActionTransitionMatrix[:, :, :, 0],
#                 agent_param=params["agent"], env_param=params["env"])


# # Frank wolfe

# visu.iter = -1


# if params["common"]["init"] == "stochastic":
#     visu.corner = "rand"
#     dPi, dPi_action = env.random_dpi(pathMatrix, pathMatrix_s)
# else:
#     # corner_ = (params["env"]["shape"]['x']-1)*(params["env"]["shape"]['y']) - 2
#     corner_ = 0
#     epsilon = 0.0001
#     dPi = torch.zeros(node_size, H)
#     dPi = epsilon * torch.ones(node_size, H)
#     dPi[corner_] = (1-epsilon*(node_size-1)) * torch.ones(1, H)
#     dPi_action = torch.zeros(H-1, node_size, env.action_dim)
#     dPi_action[:, :, 0] = epsilon * torch.ones(H-1, node_size)
#     dPi_action[:, corner_, 0] = (1-epsilon*(node_size-1)) * torch.ones(H-1)
#     visu.corner = corner_

# dPi_init = dPi[:, 0]
# env.dPi_init = dPi_init
# # print("Going for solution", pathMatrix.shape, " ", pathMatrix_s.shape)
# # optimal = env.optimal_J_pi(pathMatrix, paths_reward, dPi_init)
# # stationary_pi_val = env.stationary_pi(dPi_init)
# # visu.stationary_pi = stationary_pi_val
# # visu.JPi_optimal = -optimal.fun
# # optimal_dPi = torch.sum(torch.mul(pathMatrix, torch.from_numpy(optimal.x).type(torch.float)), 1)[
# #     :-node_size].reshape(-1, node_size, env.action_dim)
# # val_optimal_dpi = env.get_J_pi_X_py(
# #     pathMatrix, dPi_init, paths_reward, optimal_dPi)
# # visu.JPi_dpi_asper_optimal_alpha = val_optimal_dpi
# # print("optimal", visu.JPi_optimal, " ", val_optimal_dpi, " ", optimal_dPi)
# visu.JPi_optimal = 20

# ##############
# # IMPORTANCE SAMPLING
# ##############

# # At a current policy collect trajectory samples
# # collect samples
# random_pi = torch.sum(env.Hori_ActionTransitionMatrix,
#                       0).transpose(0, 2).transpose(1, 2)
# pi_h_s_a_old = torch.divide(random_pi, torch.sum(random_pi, 2)[:, :, None])
# pi_h_s_a = torch.divide(random_pi, torch.sum(random_pi, 2)[:, :, None])
# pi_h_s_a.requires_grad_()
# # pi_h_s_a_old = torch.divide(
# #     dPi_action, torch.sum(dPi_action, 2)[:, :, None])
# # pi_h_s_a = torch.divide(dPi_action, torch.sum(dPi_action, 2)[:, :, None])
# # pi_h_s_a[pi_h_s_a != pi_h_s_a] = 0
# # pi_h_s_a.requires_grad_()
# grad_old = torch.zeros(H-1, env.node_size, env.action_dim)
# beta = 0.7

# Agent's policy
if params["alg"]["type"]=="M" or params["alg"]["type"]=="SRL":
    agent = agent_net(2, env.action_dim)
else:
    agent = agent_net(H-1, env.action_dim)
optim = torch.optim.Adam(agent.parameters(), lr=params["alg"]["lr"])

for t_eps in range(epochs):
    mat_action = []
    mat_state = []
    mat_return = []
    marginal_return = []
    mat_done = []
    # print(t_eps)
    env.initialize()
    mat_state.append(env.state)
    init_state = env.state
    # print(torch.mean(env.weighted_traj_return([init_state])))
    # print(t_eps, " ", mat_state, " ", env.weighted_traj_return(mat_state))
    list_batch_state = []
    for h_iter in range(H-1):
        if params["alg"]["type"]=="M" or params["alg"]["type"]=="SRL":
            batch_state = mat_state[-1].reshape(-1, 1).float()
            # append time index to the state
            batch_state = torch.cat(
                [batch_state, h_iter*torch.ones_like(batch_state)], 1)
        else:
            batch_state = append_state(mat_state, H-1)
        action_prob = agent(batch_state)
        # action_prob = pi_h_s_a[h_iter, mat_state[-1]]
        # policy_dist = Categorical(torch.nn.Softmax()(action_prob))
        policy_dist = Categorical(action_prob)
        actions = policy_dist.sample()
        mat_action.append(actions)
        env.step(h_iter, actions)
        mat_state.append(env.state)  # s+1
        # print(t_eps, " ", mat_state, " ", env.weighted_traj_return(mat_state))
        # mat_return.append(env.batched_marginal_coverage(
        #     mat_state, [init_state]))
        mat_return.append(env.weighted_traj_return(mat_state, type = params["alg"]["type"]))
        if h_iter ==0:
            marginal_return.append(mat_return[h_iter])
        else:
            # if params["alg"]["type"]=="SRL":
            marginal_return.append(mat_return[h_iter])
            # else:
            # marginal_return.append(mat_return[h_iter] - mat_return[h_iter-1])
        list_batch_state.append(batch_state)
        # mat_return.append(env.weighted_traj_return(
        #     mat_state) - env.weighted_traj_return([init_state]))

    ###################
    # Compute gradients
    ###################
    # Remove the last state of the trajectory
    # states_visited = torch.vstack(mat_state)[:-1, :]
    states_visited = torch.vstack(list_batch_state).float()
    # policy_dist = Categorical(pi_h_s_a[states_visited.tolist()])
    # policy_dist = Categorical(torch.nn.Softmax()(torch.vstack(
    #     [pi_h_s_a[i, states_visited.tolist()[i]] for i in range(H-1)])))
    # policy_dist = Categorical(torch.vstack(
    #     [pi_h_s_a[i, states_visited.tolist()[i]] for i in range(H-1)]))  # working one
    policy_dist = Categorical(agent(states_visited))
    log_prob = policy_dist.log_prob(torch.hstack(mat_action))
    batch_return = torch.hstack(marginal_return)/MAX_Ret

    # - 2*policy_dist.entropy().mean()
    J_obj = -1*(torch.mean(log_prob*batch_return) + params["alg"]["ent_coef"] *
                policy_dist.entropy().mean()/(t_eps+1))
    optim.zero_grad()
    J_obj.backward()
    optim.step()

    obj = env.weighted_traj_return(mat_state).float()
    print(visu.JPi_optimal, " mean ", obj.mean(), " max ",
          obj.max(), " median ", obj.median(), " min ", obj.min(), " ent ", policy_dist.entropy().mean().detach())

    wandb.log({"opt": MAX_Ret, "mean": obj.mean(),
               "max": obj.max(), "median": obj.median(), "min ": obj.min(), " ent ": policy_dist.entropy().mean().detach()})
    # # roll out the policy 
    # path = []
    # env.initialize()
    # path.append(env.state[0].item())
    # for h_iter in range(H-1):
    #     batch_state = path[-1]
    #     batch_state = torch.Tensor([path[-1], h_iter])
    #     action_prob = agent(batch_state)
    #     # action = torch.argmax(action_prob)
    #     policy_dist = Categorical(action_prob)
    #     action = policy_dist.sample()
    #     env.step(h_iter, torch.ones_like(env.state)*action)
    #     path.append(env.state[0].item())
    # plt, fig = visu.stiener_grid( items_loc=env.items_loc, path=path, init=34)
    # wandb.log({"chart": wandb.Image(fig)})
    # grad_new = pi_h_s_a.grad

    # grad = grad_old*beta + (1-beta)*grad_new
    # grad_old = grad.clone()

    # ##################
    # # Frank Wolfe
    # ##################
    # sol_FW = policy.get_FW_vertex_pi(
    #     grad, dPi_init, pi_h_s_a.detach().numpy())
    # # sol_FW = policy.get_FW_vertex_pi_reparam(
    # #     pathMatrix, grad[:, :, 1:], dPi_init, pi_h_s_a.detach().numpy())
    # lr = 5/(5+t_eps)
    # # if fw_i > 3:
    # #     lr = 1/(1+fw_i/4)
    # pol = torch.from_numpy(sol_FW.x).type(torch.float).reshape(
    #     H-1, node_size, -1)
    # pi_h_s_a_old = pol*(lr) + (1-lr)*pi_h_s_a_old
    # # obj = (torch.stack(mat_return)
    # #        [-1] + env.weighted_traj_return([init_state])).float()
    # obj = env.weighted_traj_return(path).float()
    # if obj.max() > 14.5:
    #     a = 1
    # print(visu.JPi_optimal, " mean ", obj.mean(), " max ",
    #       obj.max(), " median ", obj.median(), " min ", obj.min(), " ent ", policy_dist.entropy().mean().detach())
    # # print(visu.JPi_optimal, " ", np.mean(
    # #     (env.weighted_traj_return([init_state]) + torch.sum(torch.stack(mat_return), 0)).tolist()))
    # visu.recordFW_SupMod(J_obj.clone().detach().numpy(), t_eps)
    # pi_h_s_a = pi_h_s_a_old.detach().clone().requires_grad_()

    a = 1
wandb.finish()
# print("Print the learnt policy")
# if params["common"]["init"] == "deterministic":
#     st = corner_
#     traj_sup_mod = []
#     traj_sup_mod.append([0, corner_])
#     for h in range(env.horizon-1):
#         print(pi_h_s_a[h, st])
#         action = torch.argmax(pi_h_s_a[h, st]).item()
#         st = torch.argmax(
#             env.Hori_ActionTransitionMatrix[:, st, action, h]).item()
#         traj_sup_mod.append([h+1, st])
#     print(traj_sup_mod)
