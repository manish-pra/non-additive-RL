# Game imports
import os
import time
import isaacgym 
import isaacgymenvs 
import wandb

import gymnasium as gym
from network import Actor, Critic
from helper import get_advantage, critic_update
import argparse
import yaml

import torch 

workspace = "subrl/gym_env"

parser = argparse.ArgumentParser(description="A foo that bars")
parser.add_argument("-param", default="gym_subrl")  # params

parser.add_argument("-env", type=int, default=0)
parser.add_argument("-i", type=int, default=8)  # initialized at origin
args = parser.parse_args()
torch.cuda.empty_cache()
# 1) Load the config file
with open(workspace + "/params/" + args.param + ".yaml") as file:
    params = yaml.load(file, Loader=yaml.FullLoader)
print(params)

# 2) Set the path and copy params from file
# exp_name = params["experiment"]["name"]
# env_load_path = workspace + "/experiments/" + datetime.today().strftime('%d-%m-%y') + \
#     datetime.today().strftime('-%A')[0:4] + \
#     "/environments/env_" + str(args.env) + "/"
env_load_path = (
    workspace + "/experiments/23-04-23-Sun/environments/env_" + str(args.env) + "/"
)
# env_load_path = workspace + "/experiments/" + "01-02-22-Tue"
save_path = env_load_path + "/" + args.param + "/"

if not os.path.exists(save_path):
    try:
        os.makedirs(save_path)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise

wandb.init(
    # set the wandb project where this run will be logged
    project="gym-PG",
    mode=params["visu"]["wb"],
    config=params,
)


# envs = gym.vector.make(params["env"]["name"], num_envs=params["env"]["num_envs"])
# envs.reset()

envs = isaacgymenvs.make(seed=0, task="Ant", num_envs=params["env"]["num_envs"], 
    sim_device="cuda:0", rl_device="cuda:0", graphics_device_id=0, headless=True, 
    multi_gpu=False, virtual_screen_capture=False,force_render=False) 
envs.is_vector_env = True 

obs_dim = envs.observation_space.shape[0]
act_dim = envs.action_space.shape[0]


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
p1 = Actor(obs_dim, act_dim, std=0.1).to(device)
q = Critic(obs_dim).to(device)

p1.load_state_dict(
    torch.load(workspace + '/experiments/23-04-23-Sun/' + 'model/agent1_1000.pth'))
q.load_state_dict(
    torch.load(workspace + '/experiments/23-04-23-Sun/' + 'model/val_1000.pth'))

optim_q = torch.optim.Adam(q.parameters(), lr=0.001)
optim_p1 = torch.optim.Adam(p1.parameters(), lr=0.005)

num_episode = 1001
traj_len = 400
curr_batch_size = params["env"]["num_envs"]

for t_eps in range(num_episode):
    # data_collection
    avg_itr = 0

    batch_mat_state = torch.empty(traj_len,params["env"]["num_envs"], obs_dim).to(device)
    batch_mat_action = torch.empty(traj_len,params["env"]["num_envs"], act_dim).to(device)
    batch_mat_reward = torch.empty(traj_len,params["env"]["num_envs"], 1).to(device)
    batch_mat_done = torch.empty(traj_len,params["env"]["num_envs"], 1).to(device)

    state = envs.reset()
    # state = torch.from_numpy(state).float().to(device)
    print(t_eps)
    # mat_state = state.clone()
    for itr in range(traj_len):
        # print(itr)
        avg_itr += 1

        with torch.no_grad():
            dist = p1(state['obs'])
            action = dist.sample().to(device)

        batch_mat_state[itr,:,:] = state['obs']
        batch_mat_action[itr,:,:] = action
        # if itr > 0:
        #     mat_state = torch.cat([mat_state.view(-1, curr_batch_size, obs_dim), 
        #                            state.view(-1, curr_batch_size, obs_dim)], dim=0)  # concate along dim = 0 
        #     mat_action = torch.cat([mat_action.view(-1, curr_batch_size, act_dim), 
        #                             action.view(-1, curr_batch_size, act_dim)], dim=0)
        # else:
        #     mat_state = state.clone()
        #     mat_action = action.clone()
        
        state, reward, done, truncated = envs.step(action)
        # state = state['obs'] #torch.from_numpy(state).float().to(device)
        # reward = torch.from_numpy(reward).float().to(device)
        # if torch.sum(terminated)>=1:
        #     a=1
        # done = terminated.clone() #torch.from_numpy(terminated).to(device)
        # print(reward)


        # reward1, done_c1 = get_reward_single(
        #     state_c1, vehicle_model.getLocalBounds(state_c1[:, 0]), 0, prev_state_c1)

        # mask_ele = ~done
        batch_mat_reward[itr,:,:] = reward.view(-1,1)
        batch_mat_done[itr,:,:] = done.view(-1,1)

        # if itr > 0:
        #     mat_reward = torch.cat([mat_reward.view(-1, curr_batch_size, 1),
        #                             reward.view(-1, curr_batch_size, 1)], dim=0)  # concate along dim = 0
        #     mat_done = torch.cat(
        #         [mat_done.view(-1, curr_batch_size, 1), done.view(-1, curr_batch_size, 1)], dim=0)
        # else:
        #     mat_reward = reward.clone().to(device)
        #     mat_done = done.clone().to(device)

        # if np.any(done.numpy()) :
        #     if batch_mat_action.nelement() == 0:
        #         batch_mat_state = mat_state.transpose(0, 1)[done].view(-1, obs_dim)
        #         batch_mat_action = mat_action.transpose(0, 1)[done].view(-1, act_dim)
        #         batch_mat_reward = mat_reward.transpose(0, 1)[done].view(-1, 1)
        #         batch_mat_done = mat_done.transpose(0, 1)[done].view(-1, 1)
        #         # progress_done1 = batch_mat_state1[batch_mat_state1.size(0) - 1, 0] - batch_mat_state1[0, 0]
        #         # progress_done2 = batch_mat_state2[batch_mat_state2.size(0) - 1, 0] - batch_mat_state2[0, 0]
        #         progress_done = torch.sum(mat_state1.transpose(0, 1)[done][:, mat_state1.size(
        #             0)-1, 0] - mat_state1.transpose(0, 1)[done][:, 0, 0])
        #         element_deducted = ~(done_c1)
        #         done_c1 = done_c1[element_deducted]
        #     else:
        #         prev_size = batch_mat_state1.size(0)
        #         batch_mat_state1 = torch.cat([batch_mat_state1, mat_state1.transpose(0, 1)[
        #                                      done].view(-1, 5)], dim=0)
        #         batch_mat_action1 = torch.cat([batch_mat_action1, mat_action1.transpose(0, 1)[
        #                                       done].view(-1, 2)], dim=0)
        #         batch_mat_reward1 = torch.cat([batch_mat_reward1, mat_reward1.transpose(0, 1)[
        #                                       done].view(-1, 1)], dim=0)
        #         batch_mat_done = torch.cat([batch_mat_done, mat_done.transpose(0, 1)[
        #                                    done].view(-1, 1)], dim=0)
        #         progress_done1 = progress_done1 + torch.sum(mat_state1.transpose(0, 1)[done][:, mat_state1.size(0) - 1, 0] -
        #                                                     mat_state1.transpose(0, 1)[done][:, 0, 0])
        #         # progress_done1 = progress_done1 + batch_mat_state1[batch_mat_state1.size(0) - 1, 0] - batch_mat_state1[prev_size, 0]
        #         # progress_done2 = progress_done2 + batch_mat_state2[batch_mat_state2.size(0) - 1, 0] - batch_mat_state2[prev_size, 0]
        #         element_deducted = ~(done_c1)
        #         done_c1 = done_c1[element_deducted]

        #     mat_state1 = mat_state1.transpose(
        #         0, 1)[remaining_xo].transpose(0, 1)
        #     mat_action1 = mat_action1.transpose(
        #         0, 1)[remaining_xo].transpose(0, 1)
        #     mat_reward1 = mat_reward1.transpose(
        #         0, 1)[remaining_xo].transpose(0, 1)
        #     mat_done = mat_done.transpose(0, 1)[remaining_xo].transpose(0, 1)

        # # print(avg_itr,remaining_xo.size(0))

        # # writer.add_scalar('Reward/agent1', reward1, t_eps)
        # itr = itr + 1

        # # or itr>900: #brak only if all elements in the array are true
        # if np.all(done.numpy()) == True:
        #     prev_size = batch_mat_state1.size(0)
        #     batch_mat_state1 = torch.cat(
        #         [batch_mat_state1, mat_state1.transpose(0, 1).reshape(-1, 5)], dim=0)
        #     batch_mat_action1 = torch.cat(
        #         [batch_mat_action1, mat_action1.transpose(0, 1).reshape(-1, 2)], dim=0)
        #     batch_mat_reward1 = torch.cat([batch_mat_reward1, mat_reward1.transpose(
        #         0, 1).reshape(-1, 1)], dim=0)  # should i create a false or true array?
        #     print("done", itr)
        #     print(mat_done.shape)
        #     # creating a true array of that shape
        #     mat_done[mat_done.size(
        #         0)-1, :, :] = torch.ones((mat_done[mat_done.size(0)-1, :, :].shape)) >= 2
        #     print(mat_done.shape, batch_mat_done.shape)
        #     if batch_mat_done.nelement() == 0:
        #         batch_mat_done = mat_done.transpose(0, 1).reshape(-1, 1)
        #         progress_done1 = 0
        #     else:
        #         batch_mat_done = torch.cat(
        #             [batch_mat_done, mat_done.transpose(0, 1).reshape(-1, 1)], dim=0)
        #     if prev_size == batch_mat_state1.size(0):
        #         progress_done1 = progress_done1
        #     else:
        #         progress_done1 = progress_done1 + torch.sum(mat_state1.transpose(0, 1)[:, mat_state1.size(0) - 1, 0] -
        #                                                     mat_state1.transpose(0, 1)[:, 0, 0])
        #     print(batch_mat_done.shape)
        #     # print("done", itr)
        #     break
    batch_mat_done[itr,:,:] = torch.ones_like(done.view(-1,1))
    # print(avg_itr)
    # val_mat_state = torch.cat([mat_state.view(-1, curr_batch_size, obs_dim), 
    #                         state.view(-1, curr_batch_size, obs_dim)], dim=0)  # concate along dim = 0 
    # mat_done[-1] = (done*0).reshape_as(mat_done[-1]) 
    # torch.cat(
                # [mat_done.view(-1, curr_batch_size, 1), mask_ele.view(-1, curr_batch_size, 1)], dim=0)
    
    # batch_val_mat_state = val_mat_state.transpose(0, 1).reshape(-1, obs_dim)
    # batch_mat_state = batch_mat_state.transpose(0, 1).reshape(-1, obs_dim)
    # batch_mat_action = batch_mat_action.transpose(0, 1).reshape(-1, act_dim)
    # batch_mat_reward = batch_mat_reward.transpose(0, 1).reshape(-1, 1)
    # batch_mat_done = batch_mat_done.transpose(0, 1).reshape(-1, 1)

    wandb.log({"reward": batch_mat_reward.mean().item()})
    # print(batch_mat_state1.shape, itr)
    # writer.add_scalar("Dist/variance_throttle_p1", dist1.variance[0, 0], t_eps)
    # writer.add_scalar("Dist/variance_steer_p1", dist1.variance[0, 1], t_eps)
    # writer.add_scalar("Reward/mean", batch_mat_reward1.mean(), t_eps)
    # writer.add_scalar("Reward/sum", batch_mat_reward1.sum(), t_eps)
    # writer.add_scalar("Progress/final_p1", progress_done1 / batch_size, t_eps)
    # writer.add_scalar("Progress/trajectory_length", itr, t_eps)
    # writer.add_scalar("Progress/agent1", batch_mat_state1[:, 0].mean(), t_eps)

    val1 = q(batch_mat_state)
    # val1 = val1.detach()
    next_value = 0  # because currently we end ony when its done which is equivalent to no next state
    returns_np1 = get_advantage(next_value, batch_mat_reward, val1.detach(), batch_mat_done, gamma=0.99, tau=0.95, device=device)
    # val2 = q(torch.stack(mat_state2))
    # val2 = val2.detach()
    # next_value = 0  # because currently we end ony when its done which is equivalent to no next state
    # returns_np2 = get_advantage(next_value, torch.stack(mat_reward2), val2, torch.stack(mat_done), gamma=0.99, tau=0.95)
    #
    # returns2 = torch.cat(returns_np2)
    # advantage_mat2 = returns2 - val2.transpose(0,1)

    # for loss_critic, gradient_norm in critic_update(batch_mat_state, torch.cat(returns_np1).view(-1, 1),  q, optim_q):
    # val_loc = q(state_mb)
    critic_loss = (returns_np1 - val1).pow(2).mean()

    optim_q.zero_grad()
    critic_loss.backward()
    optim_q.step()
    wandb.log({"critic_loss": critic_loss.item()})
    
    advantage_mat1 = (returns_np1 - val1).detach()[:,:,0]
    del returns_np1
        # print('critic_update')
    # ed_q_time = time.time()

    wandb.log({"advantage": advantage_mat1.mean()})
    # st_time = time.time()
    # calculate gradients
    dist_batch1 = p1(batch_mat_state)
    log_probs1_inid = dist_batch1.log_prob(batch_mat_action)
    log_probs1 = log_probs1_inid.sum(-1)

    optim_p1.zero_grad()
    loss = -(log_probs1 * advantage_mat1).mean() - 0.01 * dist_batch1.entropy().mean()
    loss.backward()
    optim_p1.step()
    # improve1, improve2, lamda, lam1, lam2, esp, stat = optim.step(
    #     advantage_mat1, batch_mat_state1, state_gpu_p2, batch_mat_action1, batch_mat_action2)

    if t_eps % 100 == 0:
            torch.save(p1.state_dict(),
                       workspace + '/experiments/23-04-23-Sun/' + 'model/agent1_' + str(
                           1000+t_eps) + ".pth")
            torch.save(q.state_dict(),
                       workspace + '/experiments/23-04-23-Sun/' + 'model/val_' + str(
                           1000+t_eps) + ".pth")