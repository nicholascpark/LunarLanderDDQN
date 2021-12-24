import copy
import numpy as np
import torch
import torch.nn as nn
from deepmodel import fclayer
from experience_replay import ReplayMemory
import sys

class DQNAgent:

    def __init__(self, state_size = 8, action_size = 4, batch_size = 32, buffer_size = 100000, epsilon_decay = 0.999, gamma = 0.99, hidden_size1 = 32, hidden_size2 = 32, hidden_size3 = 0, learning_rate = 0.0001, regularization = 0.000001, momentum = 0.99, qnet_update_freq = 5, double_dqn = True):

        self.state_size = state_size
        self.action_size = action_size
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        print("Currently using:", self.device)
        self.double_dqn = double_dqn

        self.hidden_size1 = hidden_size1
        self.hidden_size2 = hidden_size2
        self.hidden_size3 = hidden_size3
        if self.hidden_size3 > 0:
            self.qnet =  self.qnet = fclayer.FourLayerNet(self.state_size, self.hidden_size1, self.hidden_size2, self.hidden_size3, self.action_size).to(self.device)
            print("DQN architecture layers", self.state_size, "->", self.hidden_size1, "->", self.hidden_size2, "->", self.hidden_size3, "->",
                  self.action_size)
        else:
            self.qnet = fclayer.ThreeLayerNet(self.state_size, self.hidden_size1, self.hidden_size2, self.action_size).to(self.device)
            print("DQN architecture layers", self.state_size, "->", self.hidden_size1, "->", self.hidden_size2, "->",
                  self.action_size)

        self.fixed_qnet = copy.deepcopy(self.qnet)
        self.qnet_update_freq = qnet_update_freq

        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.reg = regularization
        self.momentum = momentum
        # self.optimizer = torch.optim.SGD(self.qnet.parameters(), lr = self.learning_rate,) #momentum = self.momentum, weight_decay = self.reg)
        # self.optimizer = torch.optim.Adam(self.qnet.parameters())
        self.optimizer = torch.optim.Adam(self.qnet.parameters(),
                                          lr = self.learning_rate, weight_decay = self.reg,
                                          betas = [0.9,0.999],
                                          eps = 1e-04, amsgrad= True)
        self.loss_fn = nn.MSELoss()

        self.buffer_size = buffer_size
        self.replay_memory = ReplayMemory(self.buffer_size)
        self.num_steps = 0
        self.gamma = gamma
        self.epsilon = 1
        self.epsilon_decay = epsilon_decay
        self.min_epsilon = 0.1

    def learn(self, sars):

        self.replay_memory.add(sars)
        if self.num_steps >= self.batch_size:
            s_sample, a_sample, r_sample, s_prime_sample, d_sample = self.replay_memory.sample(self.batch_size)
            if torch.cuda.is_available():
                s_sample = s_sample.cuda()
                s_prime_sample = s_prime_sample.cuda()
            Q_s_prime_a_prime = self.fixed_qnet(s_prime_sample)
            if self.double_dqn:
                argmax_a_prime = self.qnet(s_prime_sample).argmax(dim=1)
                max_Q_s_prime_a_prime = Q_s_prime_a_prime[np.arange(self.batch_size), argmax_a_prime].unsqueeze(1)
            else:
                max_Q_s_prime_a_prime = Q_s_prime_a_prime.max(1)[0].unsqueeze(1)
            targets = r_sample + self.gamma*max_Q_s_prime_a_prime*(1-d_sample)
            Q_s_a = self.qnet(s_sample)
            q_values = Q_s_a[np.arange(self.batch_size), a_sample].unsqueeze(1)
            # q_values = torch.gather(Q_s_a, 1, a_sample)
            self.optimizer.zero_grad()
            loss = self.loss_fn(q_values, targets)
            loss.backward()
            self.optimizer.step()

        if self.num_steps % self.qnet_update_freq == 0:
            self.fixed_qnet = copy.deepcopy(self.qnet)

        self.num_steps += 1

    def pick_action(self, state):

        # action = np.random.randint(self.action_size)
        # print(self.epsilon)
        if np.random.random() < self.epsilon:
            action = np.random.randint(self.action_size)
        else:
            state = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
            self.qnet.eval()
            with torch.no_grad():
                q_values = self.qnet(state)
            self.qnet.train()
            action = np.argmax(q_values.cpu().data.numpy())
            # print(action)

        return action

    # def softmax(self, action_values, tau=0.01):

    #     max_action_value = torch.max(action_values, axis=1, keepdim=True)[0] / tau
    #     action_values = action_values / tau
    #     preference = action_values - max_action_value
    #     exp_action = torch.exp(preference)
    #     sum_exp_action = torch.sum(exp_action, axis=1).view(-1, 1)
    #     probs = exp_action / sum_exp_action
    #     return probs



