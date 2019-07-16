import torch
import random
import sys

class Env:
    def __init__(self, digit_num=4, success_reward=1, action_cost=0):
        self.digit_num = digit_num    # digit num
        self.success_reward = success_reward
        self.action_cost = action_cost

        self.digits = []
        self.labels = []
        self.state = []
        self.correct_state = []

        self.actions = [(i, i+1) for i in range(self.digit_num-1)]

    def new_setting(self, digits, labels):
        self.digits = [digits.squeeze()[i] for i in range(self.digit_num)]
        self.labels = [labels[i].item() for i in range(self.digit_num)]
        self.state = [labels[i].item() for i in range(self.digit_num)]
        self.state_order = [i for i in range(self.digit_num)]
        self.correct_state = sorted(self.state)

        if self.state == [labels[0] for _ in range(self.digit_num)]:   # ゾロ目の場合
            return True
        error_count = 0
        while self._is_sorted():    # 初期状態で揃っていた場合，並び替える
            error_count += 1
            change_num = random.randint(1, self.digit_num*(self.digit_num-1)//2)    # 並び替える回数
            for _ in range(change_num):
                action = self.actions[random.randint(0, self.digit_num-2)]
                self._exchange(action)
            if error_count > 1000:
                print('REPDIGIT ERROR')
                return True
        return False

    def _exchange(self, action):
        loc1, loc2 = action
        self.state[loc1], self.state[loc2] = self.state[loc2], self.state[loc1]
        self.state_order[loc1], self.state_order[loc2] = self.state_order[loc2], self.state_order[loc1]

    def _is_sorted(self):
        return self.state == self.correct_state

    def now_state(self, mode='mnist'):
        if mode == 'mnist':
            return torch.cat([self.digits[i].unsqueeze(0) for i in self.state_order], dim=0).unsqueeze(0)
        elif mode == 'digit':
            return torch.cat([self.digits[i].unsqueeze(0) for i in self.state_order]).view(1,-1)

    def take_action(self, action_num):
        self._exchange(self.actions[action_num])
        reward = self.action_cost
        done = False
        if self._is_sorted():   # ソート完了
            reward = self.success_reward
            done = True

        return reward, done

    # お遊び
    def mnist_print(self):      
        for i in range(28):
            for j in range(28*self.digit_num):
                if torch.cat([self.digits[i] for i in self.state_order], dim=1)[i][j].item() > 0.5:
                    print('1', end='')
                else:
                    print('0', end='')
            print()


