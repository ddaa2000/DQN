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


class ReplayMemory(object):

    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        """Saves a transition."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

class DQN(nn.Module):

    def __init__(self, h, w, outputs):
        super(DQN, self).__init__()
        #这里本来应该是4个channel即：
        self.conv1 = nn.Conv2d(4, 16, kernel_size=8, stride=4)
        #但是由于现在state只包含了一帧所以现在这样写以免报错：
        #self.conv1 = nn.Conv2d(1, 16, kernel_size=8, stride=4)
       # self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=4, stride=2)
       # self.bn2 = nn.BatchNorm2d(32)

       # self.bn3 = nn.BatchNorm2d(32)

        # Number of Linear input connections depends on output of conv2d layers
        # and therefore the input image size, so compute it.
        def conv2d_size_out1(size, kernel_size=8, stride=4):
            return (size - (kernel_size - 1) - 1) // stride + 1

        def conv2d_size_out2(size, kernel_size = 4, stride=2):
            return (size - (kernel_size - 1) - 1) // stride + 1

        convw = conv2d_size_out2(conv2d_size_out1(w))
        convh = conv2d_size_out2(conv2d_size_out1(h))
        linear_input_size = convw * convh * 32
        self.fullyConnected = nn.Linear(linear_input_size, 256)
        self.head = nn.Linear(256, outputs)

    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.fullyConnected(x.view(x.size(0), -1)))
        return self.head(x.view(x.size(0), -1))

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
print(get_screen())
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

policy_net = DQN(screen_height, screen_width, n_actions).to(device)
target_net = DQN(screen_height, screen_width, n_actions).to(device)
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()

optimizer = optim.RMSprop(policy_net.parameters())
memory = ReplayMemory(10000)


steps_done = 0


def select_action(state):
    global steps_done
    sample = random.random()
    #eps_threshold = EPS_END + (EPS_START - EPS_END) * \
    #   math.exp(-1. * steps_done / EPS_DECAY)
    #论文中从1到0.1在1000000轮内线性下降，1000000轮后保持0.1
    if steps_done <= 1000000:
        eps_threshold = EPS_START - (EPS_START-EPS_END) * steps_done/1000000
    else:
        eps_threshold = 0.1
    steps_done += 1
    if sample > eps_threshold:
        with torch.no_grad():
            # t.max(1) will return largest column value of each row.
            # second column on max result is index of where max element was
            # found, so we pick action with the larger expected reward.
            return policy_net(state).max(1)[1].view(1, 1)
    else:
        return torch.tensor([[random.randrange(n_actions)]], device=device, dtype=torch.long)


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

def optimize_model():
    if len(memory) < BATCH_SIZE:
        return
    transitions = memory.sample(BATCH_SIZE)
    # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
    # detailed explanation). This converts batch-array of Transitions
    # to Transition of batch-arrays.
    batch = Transition(*zip(*transitions))

    # Compute a mask of non-final states and concatenate the batch elements
    # (a final state would've been the one after which simulation ended)
    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                          batch.next_state)), device=device, dtype=torch.bool)
    non_final_next_states = torch.cat([s for s in batch.next_state
                                                if s is not None])
    state_batch = torch.cat(batch.state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)
    #print(state_batch.shape)

    # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
    # columns of actions taken. These are the actions which would've been taken
    # for each batch state according to policy_net
    state_action_values = policy_net(state_batch).gather(1, action_batch)

    # Compute V(s_{t+1}) for all next states.
    # Expected values of actions for non_final_next_states are computed based
    # on the "older" target_net; selecting their best reward with max(1)[0].
    # This is merged based on the mask, such that we'll have either the expected
    # state value or 0 in case the state was final.
    next_state_values = torch.zeros(BATCH_SIZE, device=device)
    next_state_values[non_final_mask] = target_net(non_final_next_states).max(1)[0].detach()
    # Compute the expected Q values
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch

    # Compute Huber loss
    #与论文中定义的损失函数不同，论文里是均方误差
    loss = F.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))

    # Optimize the model
    optimizer.zero_grad()
    loss.backward()
    for param in policy_net.parameters():
        param.grad.data.clamp_(-1, 1)
    optimizer.step()

num_episodes = 50
for i_episode in range(num_episodes):
    # Initialize the environment and state
    env.reset()
    #last_screen = get_screen()
    current_screen = get_screen()
    #state = current_screen - last_screen
    # 关于最开始几帧的处理，如果未能截取够四帧，剩下的由零值填补。不知道这样可不可行？
    s = current_screen
    zero = torch.zeros(s.shape)
    state = torch.cat((s,zero,zero,zero),dim=1)
    # print('state_all shape:',state_all.shape)
    #print('state_all:', state_all)
    for t in count(): # count()作用：生成无限序列，从0开始，只有通过显示中断操作使其退出循环，否则一直循环下去
        # Select and perform an action

        action = select_action(state)#选择动作

        plt.figure(1)

        # 强行改了一下，使得图像变成了110*84，但是论文上要的是84*84，需要再改一下
        # 在resize上改动后，图像变为84*84
        #print(get_screen().shape)
        plt.imshow(env.render(mode='rgb_array'),
                   interpolation='none')
        _, reward, done, _ = env.step(action.item())#执行动作
        plt.pause(0.001)
        reward = torch.tensor([reward], device=device)#执行动作之后得到的奖励

        # Observe new state
        #last_screen = current_screen
        current_screen = get_screen()
        if not done:
            next_state = torch.cat((current_screen,state[0][0].unsqueeze(0).unsqueeze(0),state[0][1].unsqueeze(0).unsqueeze(0),state[0][2].unsqueeze(0).unsqueeze(0)),dim=1)
        else:
            next_state = None
        #print(next_state)
        #print(next_state.shape)
        # Store the transition in memory

        memory.push(state, action, next_state, reward)

        # Move to the next state
        state = next_state


        # Perform one step of the optimization (on the target network)
        optimize_model()
        if done:
            episode_durations.append(t + 1)
            plot_durations()
            break
    # Update the target network, copying all weights and biases in DQN
    if i_episode % TARGET_UPDATE == 0:
        target_net.load_state_dict(policy_net.state_dict())

print('Complete')
env.render()
env.close()
plt.ioff()
plt.show()