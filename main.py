import torch

import dqn

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 
    print(device)

    chosen_model = 'mnist'
    model_dict = {'digit':dqn.DQN_DIGIT(), 'mnist':dqn.DQN_MNIST()}

    for g in [0.8, 0.85, 0.9, 0.95]:
        for t in [10, 30, 50, 100]:
            for b in [10, 50, 100, 500]:
                for f in [0, -1]:
                    print('====================')
                    print('gamma:{}, target_update:{}, batch:{}, failure_reward:{}'.format(g, t, b, f))
                    learning = dqn.Learning(device, model_dict[chosen_model], learning_mode=chosen_model, gamma=g, target_update=t, batch_size=b, failure_reward=f)
                    learning.training(EPOCH=5)
                    