import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import random
from collections import namedtuple
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import time
import os

import env

class DQN_DIGIT(nn.Module):
    def __init__(self):
        super(DQN_DIGIT, self).__init__()
        self.f1 = nn.Linear(4, 100)
        self.f2 = nn.Linear(100, 100)
        self.f3 = nn.Linear(100, 4)
        
    def forward(self, x):
        x = F.relu(self.f1(x))
        x = F.relu(self.f2(x))
        x = F.relu(self.f2(x))
        x = F.relu(self.f3(x))
        return x

class DQN_MNIST(nn.Module):
    def __init__(self):
        super(DQN_MNIST, self).__init__()
        self.conv1 = nn.Conv2d(1, 20, 5, 1)
        self.conv2 = nn.Conv2d(20, 50, 5, 1)
        self.fc1 = nn.Linear(5000, 500)
        self.fc2 = nn.Linear(500, 4)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 5000)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

class ReplayMemory(object):
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0
        self.transition = namedtuple('Transition', 
            ('state', 'action', 'next_state', 'reward'))
    
    def push(self, *args):
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = self.transition(*args)
        self.position = (self.position + 1) % self.capacity
    
    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)
    
    def __len__(self):
        return len(self.memory)

class Learning:
    def __init__(self, device, model, learning_mode='mnist', seed=0, 
        state_size=4, batch_size=128, gamma=0.9, target_update=10, 
        replay_memory_size=10**4, success_reward=1, failure_reward=0):
        self.device = device
        self.state_size = state_size
        self.batch_size = batch_size
        self.gamma = gamma
        self.target_update = target_update

        self.success_reward = success_reward
        self.failure_reward = failure_reward
        self.env = env.Env(success_reward=self.success_reward, failure_reward=self.failure_reward)
        self.replay_memory_size = replay_memory_size
        self.memory = ReplayMemory(self.replay_memory_size)

        self.policy_net = model.to(self.device)
        self.target_net = model.to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        
        self.optimizer = optim.Adam(self.policy_net.parameters())

        self.learning_mode = learning_mode
        self.seed = seed
        if self.learning_mode == 'mnist':
            # train data : 60000
            # test data : 10000
            train_data = torchvision.datasets.MNIST('.', train=True, 
                download=True, transform=transforms.ToTensor())
            test_data = torchvision.datasets.MNIST('.', train=False, 
                download=True, transform=transforms.ToTensor())
        elif self.learning_mode == 'digit':
            np.random.seed(self.seed)
            x = torch.from_numpy(np.random.randint(0,10, 60000)).float()
            train_data = TensorDataset(x, x)
            y = torch.from_numpy(np.random.randint(0,10, 10000)).float()
            test_data = TensorDataset(y, y)


        self.train_loader = DataLoader(train_data, batch_size=self.state_size)
        self.test_loader = DataLoader(test_data, batch_size=self.state_size)
    
        self.dirpath = './'


    def select_action(self, state, episode):
        sample = random.random()
        EPS_START = 0.9
        EPS_END = 0.05
        EPS_DECAY = 20000
        eps_threshold = EPS_END + (EPS_START - EPS_END) * np.exp(-1. * episode / EPS_DECAY)

        if sample > eps_threshold:
            with torch.no_grad():
                return self.policy_net(state).argmax().view(1, 1)
        else:
            return torch.tensor([[random.randrange(self.state_size)]], 
                device=self.device, dtype=torch.long)

    def optimize_model(self):
        if len(self.memory) < self.batch_size:
            return
        
        transitions = self.memory.sample(self.batch_size)

        batch = self.memory.transition(*zip(*transitions))
    
        non_final_mask = torch.tensor(tuple(map(lambda s:s is not None, 
            batch.next_state)), device=self.device, dtype=torch.long)
        non_final_next_states = torch.cat([s for s in batch.next_state 
            if s is not None])
        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)
        
        state_action_values = self.policy_net(state_batch).gather(1, action_batch)

        next_state_values = torch.zeros(self.batch_size, device=self.device)
        next_state_values[non_final_mask] = self.target_net(non_final_next_states).max().detach()

        expected_state_action_values = (next_state_values * self.gamma) + reward_batch.float()

        loss = F.mse_loss(state_action_values, expected_state_action_values.unsqueeze(1)) 

        self.optimizer.zero_grad()
        loss.backward()

        self.optimizer.step()


    def training(self, EPOCH=1):
        self.make_dir()
        start_time = time.time()
        step_average = [[np.nan],[np.nan]]
        success_average = [[np.nan],[np.nan]]
        total_episode = 0
        for epo in range(EPOCH):
            step_sum = 0
            success_sum = 0
            for episode, (digits, labels) in enumerate(self.train_loader):
                digits = digits.to(self.device)
                self.env.new_setting(digits, labels)
                state = self.env.now_state(self.learning_mode)

                done = False
                step = 0
                while not done:
                    step += 1

                    action = self.select_action(state, total_episode)
                    if step == 20:  # ソートが長くなりすぎる
                        action = torch.tensor([[self.state_size-1]], 
                            device=self.device, dtype=torch.long)



                    reward, done = self.env.take_action(action.item())
                    r = reward
                    reward = torch.tensor([reward], device=self.device)

                    if not done:
                        next_state = self.env.now_state(self.learning_mode)
                    else:
                        next_state = None

                    self.memory.push(state, action, next_state, reward)
                    state = next_state

                    self.optimize_model()

                step_sum += step
                success = 0
                if r == 1:
                    success = 1
                    success_sum += 1
                total_episode = epo * len(self.train_loader) + episode + 1
                if total_episode % self.target_update == 0:
                    self.target_net.load_state_dict(self.policy_net.state_dict())
                if total_episode % 100 == 0:
                    step_average[0].append(total_episode)
                    step_average[1].append(step_sum/100)
                    success_average[0].append(total_episode)
                    success_average[1].append(success_sum/100)
                    step_sum = 0
                    success_sum = 0
                if total_episode % 1000 == 0:
                    self.plot_step(step_average, success_average, self.dirpath+'plot.png')
                if total_episode % 10000 == 0:
                    self.save_model(total_episode, self.dirpath+'model{}.pth'.format(total_episode))
                
                # print('eposode:{}, success={}, steps:{}'.format(total_episode, success, step))
            
        print('total time = {} mins'.format((time.time()-start_time)//60))

    def plot_step(self, step_average, success_average, path):
        _, ax1 = plt.subplots()
        ax1.plot(step_average[0], step_average[1], label='step')
        ax2 = ax1.twinx()
        ax2.plot(success_average[0], success_average[1], label='success rate', color='r')
        ax1.set_ylabel("step")
        ax2.set_ylabel("success rate")
        ax1.legend(loc='upper left')
        ax2.legend(loc='upper center')
        plt.savefig(path)
        plt.close()
    
    def save_model(self, episode, path):
        checkpoint = {
            'episode': episode,
            'model_state': self.target_net.state_dict(),
            'optimizer_state': self.optimizer.state_dict(),
            'learning_mode': self.learning_mode,
            'batch_size': self.batch_size,
            'seed': self.seed,
            'gamma': self.gamma,
            'target_update': self.target_update,
            'replay_memory_size': self.replay_memory_size,
            'success_reward': self.success_reward,
            'failure_reward': self.failure_reward
        }
        torch.save(checkpoint, path)

    def make_dir(self, dirname='result/result'):
        dirnum = 1
        while os.path.exists(dirname+str(dirnum)):
            dirnum += 1
        self.dirpath = './{}{}/'.format(dirname, dirnum)
        os.mkdir(self.dirpath)

        with open(self.dirpath+'setting({}).txt'.format(self.learning_mode), 'w') as f:
            f.write('mode:{}\nbatch size: {}\nseed: {}\ngamma: {}\ntarget_update: {}\nreplay_memory_size: {}\nsuccess_reward: {}\nfailure_reward: {}\n'
                .format(self.learning_mode, self.batch_size, self.seed, self.gamma, 
                self.target_update, self.replay_memory_size, self.success_reward, 
                self.failure_reward))