import torch

class Env:
    def __init__(self, state_size=4, success_reward=1, failure_reward=0, take_action_reward=0):
        self.state_size = state_size
        self.success_reward = success_reward
        self.failure_reward = failure_reward
        self.take_action_reward = take_action_reward

        self.digits = []
        self.labels = []
        self.state = []
        self.correct_state = []

        self.actions = [(0,1), (1,2), (2,3), None]

    def new_setting(self, digits, labels):
        self.digits = [digits.squeeze()[i] for i in range(self.state_size)]
        self.labels = [labels[i].item() for i in range(self.state_size)]
        self.state = [labels[i].item() for i in range(self.state_size)]
        self.state_order = [i for i in range(self.state_size)]
        self.correct_state = sorted(self.state)

    def _terminate(self):
        reward = self.failure_reward
        if self._is_sorted():
            reward = self.success_reward
        return reward

    def _exchange(self, loc1, loc2):
        self.state[loc1], self.state[loc2] = self.state[loc2], self.state[loc1]
        self.state_order[loc1], self.state_order[loc2] = self.state_order[loc2], self.state_order[loc1]

    def _is_sorted(self):
        return self.state == self.correct_state

    def now_state(self, mode='mnist'):
        if mode == 'mnist':
            return torch.cat([self.digits[i] for i in self.state_order], dim=1).view(1,1,28,112)
        elif mode == 'digit':
            return torch.cat([self.digits[i].unsqueeze(0) for i in self.state_order]).view(1,-1)
    def take_action(self, action_num):
        done = False
        if self.actions[action_num] is None:
            done = True
            reward = self._terminate()
            return reward, done
        else:
            loc1, loc2 = self.actions[action_num]
            self._exchange(loc1, loc2)
            return 0, done
        
    def mnist_print(self):      # お遊び
        for i in range(28):
            for j in range(112):
                if torch.cat([self.digits[i] for i in self.state_order], dim=1)[i][j].item() > 0.5:
                    print('1', end='')
                else:
                    print('0', end='')
            print()


