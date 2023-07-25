import gymnasium as gym

from gymnasium.envs.mujoco import ant_v4
import torch
from network import Actor, Critic
import numpy as np

class MyAntEnv(ant_v4.AntEnv):
    def __init__(self,rew_type, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.mask = torch.zeros(100,100)
        self.traj_rew = 0
        self.rew_type = rew_type
    
    def step(self, action):
        observation, reward, terminated, truncated, info = super().step(action)

        x_pos = observation[0]
        y_pos = observation[1]       
        x_arg = int((x_pos+20)/(2*20/100))
        y_arg = int((y_pos+20)/(40/100))
        self.mask[x_arg:5+x_arg, y_arg:5+y_arg] = torch.ones_like(self.mask[x_arg:5+x_arg, y_arg:5+y_arg])
        if self.rew_type=="modular":
            reward = (torch.sum(self.mask)).item()/10.0
        else:
            reward = (torch.sum(self.mask) - self.traj_rew ).item()/10.0
        self.traj_rew = torch.sum(self.mask)
        info['coverage'] = self.traj_rew.item()
        # if abs(info["x_position"]) > 20 or abs(info["y_position"]) > 10:
        if abs(observation[0]) > 20 or abs(observation[1]) > 20:
            super().reset()
            self.mask = torch.zeros(100,100)
            terminated = True
        return observation, reward, terminated, False, info 
    
    def reset(self):
        obs, info = super().reset()
        self.mask = torch.zeros(100,100)
        x_pos = obs[0]
        y_pos = obs[1]       
        x_arg = int((x_pos+20)/(2*20/100))
        y_arg = int((y_pos+20)/(40/100))
        self.mask[x_arg:5+x_arg, y_arg:5+y_arg] = torch.ones_like(self.mask[x_arg:5+x_arg, y_arg:5+y_arg])
        self.traj_rew = torch.sum(self.mask)
        info['coverage'] = self.traj_rew.item()
        return obs, info
        
if __name__=="__main__":
    # envs = gym.vector.AsyncVectorEnv([MyAntEnv]*3)
    envs = gym.vector.AsyncVectorEnv([lambda: MyAntEnv(rew_type="submodular",exclude_current_positions_from_observation=False, render_mode="human")])
    state, info = envs.reset()
    state = torch.from_numpy(state).float()

    workspace = "subrl/gym_env"

    device = "cpu"
    p1 = Actor(envs.observation_space.shape[1], envs.action_space.shape[1], std=0.1).to(device)
    # p1 = Actor(envs.observation_space.shape[1]-2, envs.action_space.shape[1], std=0.1).to(device)
    # p1.load_state_dict(
    #     torch.load(workspace + '/experiments/22-04-23-Sat/' + 'model/agent1_900.pth'))
    p1.load_state_dict(
        torch.load(workspace + '/experiments/12-05-23-Fri/environments/gym_subRL/' + 'agent1_20000.pth'))

    ang_list = []
    for i in range(800):
        # dist = p1(state[:,2:])
        dist = p1(state)
        action = dist.mean.detach().numpy()#().to('cpu')
        # # action = action*np.array([1, 1, 0, 0, 0, 0, 0, 0])
        # if i%2==0:
        #     action = np.array([0, 0, 0, 0, 0, 0, 0, 1]).reshape(-1,8)
        # else:
        #     action = np.array([0, 0, 0, 0, 0, 0, 0, -1]).reshape(-1,8)
        # action = (
        #     envs.action_space.sample()
        # )  # agent policy that uses the observation and info
        state, reward, terminated, truncated, info = envs.step(action)
        state = torch.from_numpy(state).float()
        print(info["coverage"])
        # angles = quaternion_to_euler_angle_vectorized2(state[:,3:7])
        # ang_list.append(angles)
        # print(angles)
        # print(np.rad2deg(angles))
        # print(reward)
        # if np.any(terminated):
        #     quit()
        #     print(terminated)
            # a=1

    # env = MyAntEnv()
    # obs = env.reset()
    # done = False
    # while not done:
    #     action = env.action_space.sample()
    #     obs, reward, done, truncated, info = env.step(action)
    #     print(obs)