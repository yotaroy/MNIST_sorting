import torch

import dqn

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 
    print(device)

    digit = 4
    chosen_model = 'mnist'
    model_dict = {'digit':dqn.DQN_DIGIT(digit), 'mnist':dqn.DQN_MNIST(digit)}

    g, t, b, a = 0.7, 10, 512, 0.0
    print('====================')
    print('gamma:{}, target_update:{}, batch:{}, action_cost:{}'.format(g, t, b, a))
    learning = dqn.Learning(device, model_dict[chosen_model], digit_num=digit, 
        learning_mode=chosen_model, gamma=g, target_update=t, 
        batch_size=b, action_cost=a)
    learning.training(EPOCH=500)
                    