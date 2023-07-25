import torch
import numpy as np

def critic_update(state_mb, return_mb, q, optim_q):
    #for k in range(num_ppo_epochs):
        #for state_mb, return_mb in get_minibatch(state_mat, returns, size_mini_batch):
    val_loc = q(state_mb)
    critic_loss = (return_mb - val_loc).pow(2).mean()

    # for param in q.critic.parameters():
    #     critic_loss += param.pow(2).sum() * 1e-3

    #loss = 0.5 * critic_loss
    #print(loss.detach().numpy(),critic_loss.detach().numpy(),actor_loss,entropy)
    optim_q.zero_grad()
    critic_loss.backward()
    optim_q.step()
    #print('updated')

    #torch.nn.utils.clip_grad_norm_(actor_critic.parameters(), 40)
    #
    # total_norm = 0
    # for p in q.parameters():
    #     param_norm = p.grad.data.norm(2)
    #     total_norm += param_norm.item() ** 2
    # total_norm = total_norm ** (1. / 2)

    del val_loc
    critic_loss_numpy = critic_loss.detach().cpu().numpy()
    del critic_loss

    yield critic_loss_numpy, 1#total_norm


# print(advantage_mat.size())
# ppo_update(state_mat, action_mat, log_probs_mat, returns, advantage_mat, clip_param=0.2)

def get_advantage(next_value, reward_mat, value_mat, masks, gamma=0.99, tau=0.95, device='cpu'):
    traj_len = reward_mat.shape[0]
    envs_num = reward_mat.shape[1]
    value_mat = torch.cat([value_mat, torch.zeros(1,envs_num,1).to(device)])
    gae = 0
    returns = torch.empty(traj_len, envs_num,1).to(device)
    for step in reversed(range(traj_len)):
        delta = reward_mat[step] + gamma * value_mat[step + 1] * (1-masks[step]) - value_mat[step]
        gae = delta + gamma * tau * (1-masks[step]) * gae
        returns[step] = gae + value_mat[step]
        # returns.insert(0, gae + value_mat[step])
    return returns

def quaternion_to_euler_angle_vectorized2(quat):
    x = quat[:,0]
    y = quat[:,1]
    z = quat[:,2]
    w = quat[:,3]
    
    ysqr = y * y

    t0 = +2.0 * (w * x + y * z)
    t1 = +1.0 - 2.0 * (x * x + ysqr)
    X = np.degrees(np.arctan2(t0, t1))

    t2 = +2.0 * (w * y - z * x)

    t2 = np.clip(t2, a_min=-1.0, a_max=1.0)
    Y = np.degrees(np.arcsin(t2))

    t3 = +2.0 * (w * z + x * y)
    t4 = +1.0 - 2.0 * (ysqr + z * z)
    Z = np.degrees(np.arctan2(t3, t4))

    return X, Y, Z