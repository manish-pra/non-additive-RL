import gym 
import isaacgym 
import isaacgymenvs 
import torch 
from network import Actor, Critic

workspace = "subrl/gym_env"

envs = isaacgymenvs.make( 
    seed=0, 
    task="Ant", 
    num_envs=200, 
    sim_device="cuda:0", 
    rl_device="cuda:0", 
    graphics_device_id=0, 
    headless=False, 
    multi_gpu=False, 
    virtual_screen_capture=False, 
    force_render=False
) 
obs_dim = envs.observation_space.shape[0]
act_dim = envs.action_space.shape[0]

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
p1 = Actor(obs_dim, act_dim, std=0.1).to(device)
q = Critic(obs_dim).to(device)
p1.load_state_dict(
    torch.load(workspace + '/experiments/23-04-23-Sun/' + 'model/agent1_1000.pth'))
envs.is_vector_env = True 
state = envs.reset() 
for _ in range(400): 
    envs.render() 
    dist = p1(state['obs'])
    action = dist.sample().to(device)
    state, reward, done, truncated = envs.step(action)