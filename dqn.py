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
import sys

import env


class DQN_DIGIT(nn.Module):
    def __init__(self, size):
        super(DQN_DIGIT, self).__init__()
        self.f1 = nn.Linear(size, 30)
        self.f2 = nn.Linear(30, 30)
        self.f3 = nn.Linear(30, size-1)
        
    def forward(self, x):
        x = F.relu(self.f1(x))
        x = F.relu(self.f2(x))
        x = F.relu(self.f2(x))
        x = F.relu(self.f3(x))
        return x


class DQN_MNIST(nn.Module):
    def __init__(self, size):
        super(DQN_MNIST, self).__init__()
        self.size = size

        self.block1 = nn.Sequential(
            nn.Conv2d(1,20,5,1),
            nn.ReLU(),
            nn.MaxPool2d(2,2),
            nn.Conv2d(20,50,5,1),
            nn.ReLU(),
            nn.MaxPool2d(2,2)
        )
        self.block2 = nn.Sequential(
            nn.Linear(4*4*50, 500), 
            nn.ReLU(),
            nn.Linear(500, 10),
            nn.ReLU()
        )
        self.fc1 = nn.Linear(10*self.size, 10*self.size)
        self.fc2 = nn.Linear(10*self.size, self.size-1)

    def forward(self, x):
        x = torch.chunk(x, self.size, dim=1)
        x = [self.block1(i) for i in x]
        x = [self.block2(i.view(-1, 4*4*50)) for i in x]
        x = torch.cat(x, dim=1)

        x = F.relu(self.fc1(x))
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
        digit_num=4, batch_size=128, gamma=0.7, target_update=5, 
        replay_memory_size=10**5, success_reward=1.0, action_cost=0.0):
        self.device = device
        self.digit_num = digit_num          # 数字の数
        self.batch_size = batch_size
        self.gamma = gamma                  # 割引率
        self.target_update = target_update  # target_networkのupdateする頻度

        self.success_reward = success_reward    # ソートが完了したときの報酬
        self.action_cost = action_cost      # 1行動ごとの即時報酬(ソートが完了する最後の行動除く)
        self.env = env.Env(digit_num=self.digit_num, 
            success_reward=self.success_reward, action_cost=self.action_cost)
        self.replay_memory_size = replay_memory_size
        self.memory = ReplayMemory(self.replay_memory_size)

        self.policy_net = model.to(self.device)
        self.target_net = model.to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        
        self.optimizer = optim.Adam(self.policy_net.parameters())

        self.learning_mode = learning_mode  # digit or mnist
        self.seed = seed    # digitデータ作成のseed

        # データの作成
        # train data: 60000, test data: 10000
        if self.learning_mode == 'mnist':
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
        self.train_loader = DataLoader(train_data, batch_size=self.digit_num, shuffle=True)
        self.test_loader = DataLoader(test_data, batch_size=self.digit_num, shuffle=False)
    
        self.dirpath = './'     # 保存するディレクトリ

    # 行動選択
    def select_action(self, state, episode):
        sample = random.random()

        EPS_START = 0.9
        EPS_END = 0.05
        EPS_DECAY = 10000 if self.learning_mode=='digit' else 100000
        eps_threshold = EPS_END + (EPS_START - EPS_END) * np.exp(-1. * episode / EPS_DECAY)

        if sample > eps_threshold:
            with torch.no_grad():
                return self.policy_net(state).argmax().view(1, 1)
        else:
            return torch.tensor([[random.randrange(self.digit_num-1)]], 
                device=self.device, dtype=torch.long)

    def optimize_model(self):
        if len(self.memory) < self.batch_size:
            return 0
        
        transitions = self.memory.sample(self.batch_size)
        batch = self.memory.transition(*zip(*transitions))
    
        error_count = 0
        while len([s for s in batch.next_state if s is not None]) == 0:
            error_count += 1
            transitions = self.memory.sample(self.batch_size)
            batch = self.memory.transition(*zip(*transitions))
            if error_count > 100:
                print('REPLAY MEMORY ERROR: SAMPLED NEXT STATES ARE ALL NONE')
                sys.exit(1)
            
        non_final_mask = torch.tensor(tuple(map(lambda s:s is not None, 
            batch.next_state)), device=self.device, dtype=torch.uint8)
        non_final_next_states = torch.cat([s for s in batch.next_state if s is not None])
        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)

        state_action_values = self.policy_net(state_batch).gather(1, action_batch)

        next_state_values = torch.zeros(self.batch_size, device=self.device)
        next_state_values[non_final_mask] = self.target_net(non_final_next_states).max(1)[0].detach()

        expected_state_action_values = (next_state_values * self.gamma) + reward_batch.float()

        loss = F.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1)) 

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()


    def training(self, EPOCH=1):
        self.make_dir()
        start_time = time.time()

        steps = []
        losses = []

        test_checks = []    # x-label of test steps and test successes
        test_steps = []
        test_successes = []

        # 移動平均
        step_hisory = []
        step_average_num = 100
        step_average = [np.nan] * (step_average_num-1)

        total_episode = 0

        test_step_average, test_success_rate = self.test_model(episode=total_episode)
        test_checks.append(total_episode)
        test_steps.append(test_step_average)
        test_successes.append(test_success_rate)

        for epo in range(EPOCH):
            for digits, labels in self.train_loader:
                if len(digits) != self.digit_num:
                    break
                step = 0
                loss = 0
                digits = digits.to(self.device)

                done = False

                error = self.env.new_setting(digits, labels)
                if error:   # 入力数字がゾロ目の場合
                    print('all the same number')
                    done = True
                else:
                    state = self.env.now_state(self.learning_mode)

                while not done:
                    step += 1

                    action = self.select_action(state, total_episode)
                    reward, done = self.env.take_action(action.item())
                    reward = torch.tensor([reward], device=self.device)

                    if not done:
                        next_state = self.env.now_state(self.learning_mode)
                    else:
                        next_state = None

                    self.memory.push(state, action, next_state, reward)
                    state = next_state

                    loss += self.optimize_model()

                if step != 0:
                    steps.append(step)
                    losses.append(loss/step if loss!=0 else np.nan)

                    step_hisory.append(step)
                    if len(step_hisory) >= step_average_num:
                        step_average.append(sum(step_hisory)/step_average_num)
                        step_hisory.pop(0)
                else:
                    steps.append(np.nan)
                    losses.append(np.nan)

                    if total_episode >= step_average_num:
                        step_average.append(np.nan)
                    
                total_episode += 1

                # target_network update
                if total_episode % self.target_update == 0:
                    self.target_net.load_state_dict(self.policy_net.state_dict())

                    # test
                    test_step_average, test_success_rate = self.test_model(episode=total_episode)
                    test_checks.append(total_episode)
                    test_steps.append(test_step_average)
                    test_successes.append(test_success_rate)
                    self.test_plot(test_checks, test_steps, test_successes, self.dirpath+'test.png')

               
                if total_episode % 100 == 0:
                    self.plot_step(steps, step_average, self.dirpath+'step.png')
                    self.plot_loss(losses, self.dirpath+'loss.png')
                
                if total_episode % 5000 == 0:
                    self.save_model(total_episode, steps, step_average, 
                        losses, test_checks, test_steps, test_successes, self.dirpath+'model{}.pth'.format(total_episode))
                
                print('episode:{}, steps:{}'.format(total_episode, step))

        print('total time = {} mins'.format((time.time()-start_time)//60))

    # step数のグラフを作る
    def plot_step(self, steps, step_average, path):
        x = range(1, len(steps)+1)
        plt.plot(x, steps, label='step', color='cornflowerblue')
        plt.plot(x, step_average, label='step_average', color='blue')
        plt.title(self.learning_mode.upper())
        plt.xlabel('episode')
        plt.ylabel('number of steps')
        plt.legend()
        plt.savefig(path)
        plt.close()

    # lossのグラフを作る
    def plot_loss(self, losses, path):
        x = range(1, len(losses)+1)
        plt.plot(x, losses, label='loss', color='cornflowerblue')
        plt.yscale('log')
        plt.title(self.learning_mode.upper())
        plt.xlabel('episode')
        plt.ylabel('loss')
        plt.legend()
        plt.savefig(path)
        plt.close()

    # モデルの保存
    def save_model(self, episode, steps, step_average, losses, test_checks, test_steps, test_successes, path):
        checkpoint = {
            'episode': episode,
            'digit_num': self.digit_num,
            'model_state': self.target_net.state_dict(),
            'optimizer_state': self.optimizer.state_dict(),
            'learning_mode': self.learning_mode,
            'batch_size': self.batch_size,
            'seed': self.seed,
            'gamma': self.gamma,
            'target_update': self.target_update,
            'replay_memory_size': self.replay_memory_size,
            'success_reward': self.success_reward,
            'action_cost': self.action_cost,
            'steps': steps,
            'step_average': step_average,
            'losses': losses,
            'test_checks': test_checks,
            'test_steps': test_steps,
            'test_successes': test_successes
        }
        torch.save(checkpoint, path)

    # 保存するdirectoryを作る，実験の条件をtxtファイルにまとめる
    def make_dir(self, dirname='result/result'):
        dirnum = 1
        while os.path.exists(dirname+str(dirnum)):
            dirnum += 1
        self.dirpath = './{}{}/'.format(dirname, dirnum)
        os.mkdir(self.dirpath)

        with open(self.dirpath+'setting({}).txt'.format(self.learning_mode), 'w') as f:
            f.write('mode: {}\nnums of digit: {}\nbatch size: {}\nseed: {}\n'
                .format(self.learning_mode, self.digit_num ,self.batch_size, self.seed))
            f.write('gamma: {}\ntarget_update: {}\n'
                .format(self.gamma,self.target_update))
            f.write('replay_memory_size: {}\nsuccess_reward: {}\naction_cost: {}\n'
                .format(self.replay_memory_size, self.success_reward, self.action_cost))

    # テストデータでの検証
    def test_model(self, path=None, episode=None, print_each_result=False):
        if path is None:
            test_network = self.target_net
        else:   
            model = torch.load(path)
            digit = model['digit_num']
            model_dict = {'digit':DQN_DIGIT(digit), 'mnist':DQN_MNIST(digit)}
            test_network = model_dict[model['learning_mode']].to(self.device)
            test_network.load_state_dict(model['model_state'])

        same = 0
        loop = 0
        ok = 0
        step_average = 0

        for num, (digits, labels) in enumerate(self.test_loader):
            if len(digits) != self.digit_num:
                break
            step = 0
            digits = digits.to(self.device)

            done = False

            error = self.env.new_setting(digits, labels)
            if error:   # 入力数字がゾロ目の場合
                if print_each_result:
                    print('#{}  all the same number'.format(num))
                same += 1
                continue

            while not done:
                step += 1
                state = self.env.now_state(self.learning_mode)
                with torch.no_grad():
                    action = test_network(state).argmax().view(1, 1)
 
                reward, done = self.env.take_action(action.item())
                reward = torch.tensor([reward], device=self.device)

                if step >= 100:
                    done = True
                    if print_each_result:
                        print('#{}  loop actions'.format(num))
                    loop += 1 

            if 0 < step < 100:
                step_average += step
                ok += 1
                if print_each_result:
                    print('#{}  step:{}'.format(num, step))

        step_average = step_average / ok if ok != 0 else 'None'
        success_rate = ok / (ok+loop)

        print('--------------------')
        print('step average:', step_average)
        print('sort success:', ok)
        print('loop error:', loop)
        print('same number error:', same)
        print('success rate:', success_rate)
        print('--------------------')

        if path is None:
            with open(self.dirpath+'test.txt', 'a') as f:
                if not episode is None:
                    f.write('#{}\n'.format(episode))
                f.write('step average:{}\n'.format(step_average))
                f.write('sort success:{}\n'.format(ok))
                f.write('loop error:{}\n'.format(loop))
                f.write('same number error:{}\n'.format(same))
                f.write('success rate:{}\n'.format(success_rate))

        return step_average, success_rate

    def test_plot(self, x, steps, successes, path):
        fig = plt.figure()
        plt.subplots_adjust(hspace=0.6)
        ax1 = fig.add_subplot(211)
        ax2 = fig.add_subplot(212)
    
        ax1.plot(x, steps)
        ax1.set_title('average steps')
        ax1.set_xlabel('episode')
        ax1.set_ylabel('step')

        ax2.plot(x, successes)
        ax2.set_title('success rate of sorting')
        ax2.set_ylabel('success rate')
        ax2.set_xlabel('episode')

        plt.savefig(path)
        plt.close()