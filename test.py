import torch
import dqn

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 
    print(device)

    model_num = 91
    model_episode = 50000 # 400000
    digit = 4

    learning = dqn.Learning(device, dqn.DQN_MNIST(digit), digit_num=digit)

    learning.test_model(path='./result/result{}/model{}.pth'.format(model_num, model_episode))
 