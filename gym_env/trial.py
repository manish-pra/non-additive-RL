import gymnasium as gym
import numpy as np
import time
from network import Actor, Critic
import torch
from helper import quaternion_to_euler_angle_vectorized2
import matplotlib.pyplot as plt
from inheritance_test import MyAntEnv

import matplotlib.pyplot as plt
from matplotlib import animation
import cv2
import numpy as np
import pyautogui
import time

def save_frames_as_gif(frames, path='./', filename='operator2.gif'):
    frames = frames[2:]
    plt.figure(figsize=(frames[0].shape[1] / 72.0, frames[0].shape[0] / 72.0), dpi=72)
    patch = plt.imshow(frames[0])
    plt.axis('off')

    def animate(i):
        patch.set_data(frames[i])

    anim = animation.FuncAnimation(plt.gcf(), animate, frames = len(frames), interval=50)
    anim.save(path + filename, fps=30)

envs = gym.vector.AsyncVectorEnv([lambda: MyAntEnv(rew_type = "submodular",exclude_current_positions_from_observation=False, render_mode="human")])

# sp = gym.spaces.Box(low=-1.0, high=2.0, shape=(29,), dtype=np.float64)
# envs = gym.vector.AsyncVectorEnv([lambda: gym.make("Ant-v4") for i in range(50)], sp)#.make("Ant-v4", num_envs=50)#, render_mode="human")
# envs = gym.make("Ant-v4")
# envs = gym.vector.make("Ant-v4", num_envs=1, render_mode="human",exclude_current_positions_from_observation=False) #, exclude_current_positions_from_observation=False
# sp = gym.spaces.Box(low=-1.0, high=2.0, shape=(27,), dtype=np.float64)
# envs.observation_space = sp
# envs.observation_space.high = 2.0
# envs.observation_space.low = -1.0
state, info = envs.reset()
state = torch.from_numpy(state).float()
# actions = np.array([1, 0, 1,1, 0, 1,1, 0, 1, 0, 1,1, 0, 1,1, 0, 1, 0, 1,1, 0, 1,1, 0]).reshape(3,8)
# observations, rewards, dones, infos, _ = envs.step(actions)
# envs.render()
workspace = "subrl/gym_env"

device = "cpu"
p1 = Actor(envs.observation_space.shape[1], envs.action_space.shape[1], std=0.1).to(device)
p1.load_state_dict(
    torch.load(workspace + '/experiments/23-04-23-Sat/' + 'model/agent1_5800.pth'))

RES = (70,70,1800,1130)
pyautogui.moveTo((7,1125))

st = time.time()
ang_list = []
total_reward = 0
frames = []
for i in range(400):
    dist = p1(state)
    action = dist.sample().numpy()#.detach().numpy()#().to('cpu')
    # img = pyautogui.screenshot(region=RES)
    # frame = np.array(img)
    # frames.append(frame)
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
    print(state[:,0], " ",state[:,1])
    angles = quaternion_to_euler_angle_vectorized2(state[:,3:7])
    ang_list.append(angles)
    # print(angles)
    print(reward, info['coverage'])
    total_reward+=reward
    # print(np.rad2deg(angles))
    # print(reward)
    # if np.any(terminated):
    #     quit()
    #     print(terminated)
        # a=1
print(total_reward)
ed = time.time()
print(ed - st)
x = [i[0] for i in ang_list]
x_ang = torch.stack(x)
y = [i[1] for i in ang_list]
y_ang = torch.stack(y)
z = [i[2] for i in ang_list]
z_ang = torch.stack(z)

plt.plot(x_ang.numpy())
plt.savefig("x")
plt.close()
plt.plot(y_ang.numpy())
plt.savefig("y")
plt.close()
plt.plot(z_ang.numpy())
plt.savefig("z")
# env = gym.make("Ant-v4", render_mode="human") #gym.make("LunarLander-v2", render_mode="human")
# observation, info = env.reset()

# for _ in range(1000):
#     action = env.action_space.sample()  # agent policy that uses the observation and info
#     observation, reward, terminated, truncated, info = env.step(action)
#     env.render()

#     if terminated or truncated:
#         observation, info = env.reset()

# env.close()
