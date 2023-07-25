import torch
import torch.autograd as autograd
import torch.nn as nn
import math
import numpy as np
from torch.distributions import Categorical

def general_conjugate_gradient(grad_x, grad_y,tot_grad_x, tot_grad_y, x_params, y_params, b, lr_x, lr_y, x=None, nsteps=10,
                               residual_tol=1e-16,
                               device=torch.device('cpu')):
    '''
    :param grad_x:
    :param grad_y:
    :param x_params:
    :param y_params:
    :param b:
    :param lr_x:
    :param lr_y:
    :param x:
    :param nsteps:
    :param residual_tol:
    :param device:
    :return: (I + sqrt(lr_x) * D_xy * lr_y * D_yx * sqrt(lr_x)) ** -1 * b

    '''
    if x is None:
        x = torch.zeros((b.shape[0],1), device=device)
    if tot_grad_x.shape != b.shape:
        raise RuntimeError('CG: hessian vector product shape mismatch')
    lr_x = lr_x.sqrt()
    r = b.clone().detach()
    p = r.clone().detach()
    rdotr = torch.dot(r.view(-1), r.view(-1))
    residual_tol = residual_tol * rdotr
    for i in range(nsteps):
        # To compute Avp
        # h_1 = Hvp_vec(grad_vec=grad_x, params=y_params, vec=lr_x * p, retain_graph=True)
        h_1 = autograd.grad(tot_grad_x, y_params, grad_outputs=lr_x*p, retain_graph=True)  # yx
        h_1 = torch.cat([g.contiguous().view(-1, 1) for g in h_1]).mul_(lr_y)
        # h_1 = Hvp_vec(grad_vec=grad_x, params=y_params, vec=lr_x * p, retain_graph=True).mul_(lr_y)
        # h_1.mul_(lr_y)
        # lr_y * D_yx * b
        # h_2 = Hvp_vec(grad_vec=grad_y, params=x_params, vec=lr_y * h_1, retain_graph=True)
        # h_2 = Hvp_vec(grad_vec=grad_y, params=x_params, vec=h_1, retain_graph=True).mul_(lr_x)
        h_2 = autograd.grad(tot_grad_y, x_params, grad_outputs=h_1, retain_graph=True)
        h_2 = torch.cat([g.contiguous().view(-1, 1) for g in h_2]).mul_(lr_x)
        # h_2.mul_(lr_x)
        # lr_x * D_xy * lr_y * D_yx * b
        Avp_ = p + h_2

        alpha = rdotr / torch.dot(p.view(-1), Avp_.view(-1))
        x.data.add_(alpha * p)
        r.data.add_(- alpha * Avp_)
        new_rdotr = torch.dot(r.view(-1), r.view(-1))
        beta = new_rdotr / rdotr
        p = r + beta * p
        rdotr = new_rdotr
        if rdotr < residual_tol:
            break
    return x, i + 1

def conjugate_gradient_trpo(grad_x_vec, grad_y_vec, kl_x_vec, kl_y_vec, lam, max_params, min_params, c, x=None, nsteps=10, residual_tol=1e-18,
                       lr=1e-3, device=torch.device('cpu')): # not able to parameters
    '''
    :param grad_x:
    :param grad_y:
    :param x_params:
    :param y_params:
    :param c: vec
    :param nsteps: max number of steps
    :param residual_tol:
    :return: A ** -1 * b

    h_1 = D_yx * p
    h_2 = D_xy * D_yx * p
    A = I + lr ** 2 * D_xy * D_yx * p
    '''
    if x is None:
        x = torch.zeros((c.shape[0],1), device=device)
    r = c.clone().detach()
    p = r.clone().detach()
    rdotr = torch.dot(r.view(-1), r.view(-1))
    residual_tol = residual_tol * rdotr
    # c = torch.cat([grad_x_vec, grad_y_vec], dim=0)
    for itr in range(nsteps):
        c1 = p[0:int(p.size(0)/2)]
        c2 = p[int(p.size(0)/2):p.size(0)]
        tot_grad_xy = autograd.grad(grad_y_vec, max_params, grad_outputs=c2, retain_graph=True)
        hvp_x_vec = torch.cat([g.contiguous().view(-1, 1) for g in tot_grad_xy]) # B1

        tot_grad_yx = autograd.grad(grad_x_vec, min_params, grad_outputs=c1, retain_graph=True)
        hvp_y_vec = torch.cat([g.contiguous().view(-1, 1) for g in tot_grad_yx]) # B2

        B = torch.cat([-hvp_x_vec,hvp_y_vec],dim=0)

        a11 = autograd.grad(kl_x_vec, max_params, grad_outputs=c1,retain_graph=True)
        a11_vec = torch.cat([g.contiguous().view(-1, 1) for g in a11]) # A11
        a22 = autograd.grad(kl_y_vec, min_params, grad_outputs=c2, retain_graph=True)
        a22_vec = torch.cat([g.contiguous().view(-1, 1) for g in a22])  # A22

        a1 = a11_vec #+ a12_vec
        a2 = a22_vec #+ a21_vec
        A = torch.cat([a1,a2],dim=0)

        # ### changed
        # A = torch.cat([c1,c2],dim=0)
        # ###delete back

        Avp_ = lam*A + B

        alpha = rdotr / torch.dot(p.view(-1), Avp_.view(-1))
        x.data.add_(alpha * p)
        r.data.add_(- alpha * Avp_)
        new_rdotr = torch.dot(r.view(-1), r.view(-1))
        beta = new_rdotr / rdotr
        p = r + beta * p
        rdotr = new_rdotr
        # print(rdotr)
        print(itr)
        if rdotr < residual_tol:
            break
    return x, itr + 1

def conjugate_gradient_2trpo(grad_vec, kl_vec, params, g, x=None, nsteps=10, residual_tol=1e-18,
                       device=torch.device('cpu')): # not able to parameters
    '''
    :param grad_x:
    :param grad_y:
    :param x_params:
    :param y_params:
    :param c: vec
    :param nsteps: max number of steps
    :param residual_tol:
    :return: A ** -1 * b

    h_1 = D_yx * p
    h_2 = D_xy * D_yx * p
    A = I + lr ** 2 * D_xy * D_yx * p
    '''
    if x is None:
        x = torch.zeros((g.shape[0],1), device=device)
    r = g.clone().detach()
    p = r.clone().detach()
    rdotr = torch.dot(r.view(-1), r.view(-1))
    residual_tol = residual_tol * rdotr
    # c = torch.cat([grad_x_vec, grad_y_vec], dim=0)
    for itr in range(nsteps):

        A = autograd.grad(kl_vec, params, grad_outputs=p,retain_graph=True)
        A_vec = torch.cat([g.contiguous().view(-1, 1) for g in A])

        Avp_ = A_vec
        # print(Avp_)

        alpha = rdotr / torch.dot(p.view(-1), Avp_.view(-1))
        x.data.add_(alpha * p)
        r.data.add_(- alpha * Avp_)
        new_rdotr = torch.dot(r.view(-1), r.view(-1))
        beta = new_rdotr / rdotr
        p = r + beta * p
        rdotr = new_rdotr
        # print(itr)
        if rdotr < residual_tol:
            break
    return x, itr + 1

def conjugate_gradient(grad_x, grad_y,  tot_grad_x, tot_grad_y, x_params, y_params, b, x=None, nsteps=10, residual_tol=1e-18,
                       lr=1e-3, device=torch.device('cpu')): # not able to parameters
    '''
    :param grad_x:
    :param grad_y:
    :param x_params:
    :param y_params:
    :param b: vec
    :param nsteps: max number of steps
    :param residual_tol:
    :return: A ** -1 * b

    h_1 = D_yx * p
    h_2 = D_xy * D_yx * p
    A = I + lr ** 2 * D_xy * D_yx * p
    '''
    if x is None:
        x = torch.zeros((b.shape[0],1), device=device)
    r = b.clone().detach()
    p = r.clone().detach()
    rdotr = torch.dot(r.view(-1), r.view(-1))
    residual_tol = residual_tol * rdotr
    for itr in range(nsteps):
        # To compute Avp
        h_1 = autograd.grad(tot_grad_x, y_params, grad_outputs=p, retain_graph=True)  # yx
        h_1 = torch.cat([g.contiguous().view(-1, 1) for g in h_1])
        h_2 = autograd.grad(tot_grad_y, x_params, grad_outputs=h_1, retain_graph=True)
        h_2 = torch.cat([g.contiguous().view(-1, 1) for g in h_2])
        Avp_ = p + lr * lr * h_2

        alpha = rdotr / torch.dot(p.view(-1), Avp_.view(-1))
        x.data.add_(alpha * p)
        r.data.add_(- alpha * Avp_)
        new_rdotr = torch.dot(r.view(-1), r.view(-1))
        beta = new_rdotr / rdotr
        p = r + beta * p
        rdotr = new_rdotr
        #print(itr)
        if rdotr < residual_tol:
            break
    return x, itr + 1

def get_minibatch(state_mat, returns, size_mini_batch):
    size_batch = state_mat.size(0)
    if size_batch < size_mini_batch:
        state_out = state_mat
        returns_out = returns
        return state_out , returns_out
    else:
        for _ in range(size_batch // size_mini_batch):
            rand_ids = np.random.randint(0, size_batch, size_mini_batch)
            yield state_mat[rand_ids, :], returns[rand_ids,:]


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

def get_advantage(next_value, reward_mat, value_mat, masks, gamma=0.99, tau=0.95):
    value_mat = torch.cat([value_mat, torch.FloatTensor([[next_value]])])
    gae = 0
    returns = []

    for step in reversed(range(len(reward_mat))):
        delta = reward_mat[step] + gamma * value_mat[step + 1] * masks[step] - value_mat[step]
        gae = delta + gamma * tau * masks[step] * gae
        returns.insert(0, gae + value_mat[step])
    return returns

def get_minibatch_act(state_mat, action_mat, log_probs, returns, advantage_mat, size_mini_batch):
    size_batch = state_mat.size(0)
    for _ in range(size_batch // size_mini_batch):
        rand_ids = np.random.randint(0, size_batch, size_mini_batch)
        yield state_mat[rand_ids, :], action_mat[rand_ids, :], log_probs[rand_ids, :], returns[rand_ids,
                                                                                       :], advantage_mat[rand_ids, :]


def actor_update(state_mat, action_mat, log_probs, returns, advantage_mat, clip_param, num_ppo_epochs, size_mini_batch,policy, optim_policy, coeff_ent):
    for k in range(num_ppo_epochs):
        for state_mb, action_mb, old_log_probs, return_mb, advantage_mb in get_minibatch_act(state_mat, action_mat,
                                                                                         log_probs, returns,
                                                                                         advantage_mat,size_mini_batch):
            pi_probs = policy(state_mb)
            dist = Categorical(pi_probs)
            entropy = dist.entropy().mean()
            new_log_probs = dist.log_prob(action_mb)
            ratio = (new_log_probs - old_log_probs).exp()

            # print(ratio,advantage_mb)


            surr1 = ratio * advantage_mb
            surr2 = torch.clamp(ratio, 1.0 - clip_param, 1.0 + clip_param) * advantage_mb

            actor_loss = - torch.min(surr1,
                                     surr2).mean()  # it just means out entire batch, and minimum  is element wise


            loss = actor_loss + coeff_ent * entropy
            #print(loss.detach().numpy(),critic_loss.detach().numpy(),actor_loss,entropy)
            optim_policy.zero_grad()
            loss.backward()

            #torch.nn.utils.clip_grad_norm_(actor_critic.parameters(), 40)

            total_norm = 0
            for p in policy.parameters():
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
            total_norm = total_norm ** (1. / 2)

            optim_policy.step()

            yield loss.detach().numpy(), actor_loss.detach().numpy(), entropy.detach().numpy(), total_norm

def get_relative_state(time_step):
    state = time_step.observations["info_state"][0]
    # print(pretty_board(time_step))
    try:
        pos = np.nonzero(state[0:20])[0]
        y = int(pos/5)
        x = int(pos%5)
        a_pos = np.array([x,y])
        a_status = 0
    except:
        try:
            pos = np.nonzero(state[20:40])[0]
            y = int(pos/5)
            x = int(pos%5)
            a_pos = np.array([x,y])
            a_status =1
        except:
            a_pos = np.array([5,1.5])
            a_status = 1
    try:
        pos = np.nonzero(state[40:60])[0]
        # print(1,pos)
        y = int(pos/5)
        x = int(pos % 5)
        b_pos = np.array([x, y])
        b_status = 0
    except:
        try: # 2nd try becasue in some cases when the game is done it is possible that none of the player exists
            pos = np.nonzero(state[60:80])[0]
            # print(2,pos)
            y = int(pos / 5)
            x = int(pos % 5)
            b_pos = np.array([x, y])
            b_status = 1
        except:
            b_pos = np.array([-1,1.5])
            b_status = 1
    if a_status==1:
        o_pos = a_pos
    elif b_status==1:
        o_pos = b_pos
    else:
        pos = np.nonzero(state[80:100])[0]
        y = int(pos / 5)
        x = int(pos % 5)
        o_pos = np.array([x, y])
    Ba = o_pos - a_pos # ball relative to a position
    Bb = o_pos - b_pos # Ball relative to b position
    G1a = np.array([4,1]) - a_pos
    G2a = np.array([4,2]) - a_pos
    G1b = np.array([0,1]) - b_pos
    G2b = np.array([0,2]) - b_pos
    rel_state = np.array([Ba,Bb, G1a, G2a, G1b, G2b]).reshape(12,)
    return a_status, b_status, rel_state

def get_two_state(time_step):
    state = time_step.observations["info_state"][0]
    # print(pretty_board(time_step))
    try:
        pos = np.nonzero(state[0:20])[0]
        y = int(pos/5)
        x = int(pos%5)
        a_pos = np.array([x,y])
        a_status = 0
    except:
        try:
            pos = np.nonzero(state[20:40])[0]
            y = int(pos/5)
            x = int(pos%5)
            a_pos = np.array([x,y])
            a_status =1
        except:
            a_pos = np.array([5,1.5])
            a_status = 1
    try:
        pos = np.nonzero(state[40:60])[0]
        # print(1,pos)
        y = int(pos/5)
        x = int(pos % 5)
        b_pos = np.array([x, y])
        b_status = 0
    except:
        try: # 2nd try becasue in some cases when the game is done it is possible that none of the player exists
            pos = np.nonzero(state[60:80])[0]
            # print(2,pos)
            y = int(pos / 5)
            x = int(pos % 5)
            b_pos = np.array([x, y])
            b_status = 1
        except:
            b_pos = np.array([-1,1.5])
            b_status = 1
    if a_status==1:
        o_pos = a_pos
    elif b_status==1:
        o_pos = b_pos
    else:
        pos = np.nonzero(state[80:100])[0]
        y = int(pos / 5)
        x = int(pos % 5)
        o_pos = np.array([x, y])
    Ba = o_pos - a_pos # ball relative to a position
    Bb = o_pos - b_pos # Ball relative to b position
    G1a = np.array([4,1]) - a_pos
    G2a = np.array([4,2]) - a_pos
    G1b = np.array([0,1]) - b_pos
    G2b = np.array([0,2]) - b_pos
    state1 = np.array([Ba, Bb, G1a, G2a, G1b, G2b]).reshape(12,)
    state2 = np.array([Bb, Ba, G1b, G2b, G1a, G2a]).reshape(12,)
    return a_status, b_status, state1, state2