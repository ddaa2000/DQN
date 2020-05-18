import gym
import math
import random
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple
from itertools import count
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T


env = gym.make('Pong-v0').unwrapped

# set up matplotlib
is_ipython = 'inline' in matplotlib.get_backend()
if is_ipython:
    from IPython import display

plt.ion()

# if gpu is to be used
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))





resize = T.Compose([T.ToPILImage(),
                    T.Resize((84,84), interpolation=Image.CUBIC),#试着在这里直接变成84*84
                    T.Grayscale(1),#这里加了一个把它变成灰度图像
                    T.ToTensor()])


def get_cart_location(screen_width):
    world_width = env.x_threshold * 2
    scale = screen_width / world_width
    return int(env.state[0] * scale + screen_width / 2.0)  # MIDDLE OF CART

def get_screen():
    # Returned screen requested by gym is 400x600x3, but is sometimes larger
    # such as 800x1200x3. Transpose it into torch order (CHW).
    screen = env.render(mode='rgb_array').transpose((2, 0, 1))
    # screen.shape=(3,210,160)
    # # Cart is in the lower half, so strip off the top and bottom of the screen
    # _, screen_height, screen_width = screen.shape
    # screen = screen[:, int(screen_height*0.4):int(screen_height * 0.8)]
    # view_width = int(screen_width * 0.6)
    # cart_location = get_cart_location(screen_width)
    # if cart_location < view_width // 2:
    #     slice_range = slice(view_width)
    # elif cart_location > (screen_width - view_width // 2):
    #     slice_range = slice(-view_width, None)
    # else:
    #     slice_range = slice(cart_location - view_width // 2,
    #                         cart_location + view_width // 2)
    # # Strip off the edges, so that we have a square image centered on a cart
    # screen = screen[:, :, slice_range]
    # # Convert to float, rescale, convert to torch tensor
    # # (this doesn't require a copy)

    #这里没改，但是好像这样出来以后灰度图像显示出来就变成全黑的了，看看怎么回事
    screen = np.ascontiguousarray(screen, dtype=np.float32) / 255
    screen = torch.from_numpy(screen)
    #print(screen.shape)
    # Resize, and add a batch dimension (BCHW)

    #注意这里应该把screen先截取成方形然后再执行下面的语句，或者也可以先不截取，这样的话神经网络的输入是84*110,不会报错，但是和论文的方形的不太一样
    #这个resize是上面定义的，就是一系列变换
    return resize(screen).unsqueeze(0).to(device)


env.reset()
plt.figure()

#强行改了一下，使得图像变成了110*84，但是论文上要的是84*84，需要再改一下
#print(get_screen())
plt.imshow(get_screen().cpu().squeeze(0).permute(1, 2, 0).repeat(1,1,3).numpy(),
           interpolation='none')
plt.title('Example extracted screen')
plt.show()

BATCH_SIZE = 32#从128调整到论文中的32
GAMMA = 0.999
EPS_START = 1#初始值和结束值调整到与论文中相同
EPS_END = 0.1
EPS_DECAY = 200
TARGET_UPDATE = 10

# Get screen size so that we can initialize layers correctly based on shape
# returned from AI gym. Typical dimensions at this point are close to 3x40x90
# which is the result of a clamped and down-scaled render buffer in get_screen()
init_screen = get_screen()
_, _, screen_height, screen_width = init_screen.shape

# Get number of actions from gym action space
n_actions = env.action_space.n




steps_done = 0





episode_durations = []


def plot_durations():
    plt.figure(2)
    plt.clf()
    durations_t = torch.tensor(episode_durations, dtype=torch.float)
    plt.title('Training...')
    plt.xlabel('Episode')
    plt.ylabel('Duration')
    plt.plot(durations_t.numpy())
    # Take 100 episode averages and plot them too
    if len(durations_t) >= 100:
        means = durations_t.unfold(0, 100, 1).mean(1).view(-1)
        means = torch.cat((torch.zeros(99), means))
        plt.plot(means.numpy())

    plt.pause(0.001)  # pause a bit so that plots are updated
    if is_ipython:
        display.clear_output(wait=True)
        display.display(plt.gcf())



num_episodes = 50
for i_episode in range(num_episodes):
    # Initialize the environment and state
    env.reset()
    current_screen = get_screen()
    s = current_screen
    zero = torch.zeros(s.shape)
    state = torch.cat((s, s, s, s), dim=1)
    k = 0
    for t in count(): # count()作用：生成无限序列，从0开始，只有通过显示中断操作使其退出循环，否则一直循环下去
        # Select and perform an action
        k = k+1
        #action = select_action(state)#选择动作
        action = torch.tensor([[random.randrange(n_actions)]], device=device, dtype=torch.long)


        _, reward, done, _ = env.step(action.item())  # 执行动作
        reward = torch.tensor([reward], device=device)#执行动作之后得到的奖励

        current_screen = get_screen()
        if not done:
            next_state = torch.cat((current_screen,state[0][0].unsqueeze(0).unsqueeze(0),state[0][1].unsqueeze(0).unsqueeze(0),state[0][2].unsqueeze(0).unsqueeze(0)),dim=1)
        else:
            next_state = None

        state = next_state
        if (k == 10):
            plt.figure(1)
            plt.imshow(get_screen().cpu().squeeze(0).permute(1, 2, 0).repeat(1, 1, 3).numpy(),
                       interpolation='none')
            plt.pause(0.001)
            k = 0
        if done:
            episode_durations.append(t + 1)
            plot_durations()
            break


print('Complete')
env.render()
env.close()
plt.ioff()
plt.show()