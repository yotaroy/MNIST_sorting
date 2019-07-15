import torch

import dqn

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 
    print(device)

    model_num = 91
    model_episode = 10000 # 400000

    learning = dqn.Learning(device, dqn.DQN_MNIST(4))

    learning.test_model(path='./result/result{}/model{}.pth'.format(model_num, model_episode))
 