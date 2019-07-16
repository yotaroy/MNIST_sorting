import torch

import dqn

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 
    print(device)

    digit = 4
    chosen_model = 'mnist'
    model_dict = {'digit':dqn.DQN_DIGIT(digit), 'mnist':dqn.DQN_MNIST(digit)}
    print('nums of digit = ', digit)

    g, t, b, a, r = 0.7, 5, 128, 0.0, 5
    print('====================')
    print('gamma:{}, target_update:{}, batch:{}, action_cost:{}, replace_memory_size:10**{}'.format(g, t, b, a, r))
    learning = dqn.Learning(device, model_dict[chosen_model], digit_num=digit, 
        learning_mode=chosen_model, gamma=g, target_update=t, 
        batch_size=b, action_cost=a, replay_memory_size=10**r)
    learning.training(EPOCH=8)
                    