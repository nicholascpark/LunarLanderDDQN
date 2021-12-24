import numpy as np
import torch
from collections import deque, namedtuple
import random

class ReplayMemory:

    def __init__(self, buffer_size = 200000):

        # self.memory_s = []
        # self.memory_a = []
        # self.memory_r = []
        # self.memory_s_prime = []
        # self.memory_d = []

        self.memory =[]
        self.buffer_size = buffer_size


        # self.memory = deque(maxlen=buffer_size)
        # self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])

        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    def length(self):

        self.lengths = np.array([len(self.memory_s), len(self.memory_a), len(self.memory_r), len(self.memory_s_prime)])
        assert np.all(self.lengths  == self.lengths[0]), "the experience memory stack is not rectangular"

        return len(self.memory_s)

    def add(self, sarsd):

        s, a, r, s_prime, d = sarsd

        if len(self.memory) >= self.buffer_size:
            del self.memory[0]

        self.memory.append((s,a,r,s_prime,d))

        # if self.length() >= self.buffer_size:
        #     self.memory_s.pop(0)
        #     self.memory_a.pop(0)
        #     self.memory_r.pop(0)
        #     self.memory_s_prime.pop(0)
        #     self.memory_d.pop(0)
        #
        # #
        # self.memory_s.append(s)
        # self.memory_a.append(a)
        # self.memory_r.append(r)
        # self.memory_s_prime.append(s_prime)
        # self.memory_d.append(d)
        # experience = self.experience(s, a, r, s_prime, d)
        # self.memory.append(experience)

    def sample(self, batch_size):

        memsize = len(self.memory)
        # print(memsize)
        idx = np.random.randint(memsize, size = batch_size)

        samples = [self.memory[i] for i in idx]

        s_sample, a_sample, r_sample, s_prime_sample, d_sample = map(list, zip(*samples))

        s_sample = torch.from_numpy(np.array(s_sample)).float().to(self.device)
        a_sample = torch.from_numpy(np.array(a_sample)).long().to(self.device)
        r_sample = torch.from_numpy(np.array(r_sample)).float().to(self.device).unsqueeze(1)
        s_prime_sample = torch.from_numpy(np.array(s_prime_sample)).float().to(self.device)
        d_sample = torch.from_numpy(np.array(d_sample)).long().to(self.device).unsqueeze(1)

        return s_sample, a_sample, r_sample, s_prime_sample, d_sample

        # s_sample = np.array(self.memory_s)[idx]
        # a_sample = np.array(self.memory_a)[idx]
        # r_sample = np.array(self.memory_r)[idx]
        # s_prime_sample = np.array(self.memory_s_prime)[idx]
        # d_sample = np.array(self.memory_d)[idx]
        #
        # s_sample = torch.from_numpy(s_sample).float().to(self.device)
        # a_sample = torch.from_numpy(a_sample).long().to(self.device)
        # r_sample = torch.from_numpy(r_sample).float().to(self.device)
        # s_prime_sample = torch.from_numpy(s_prime_sample).float().to(self.device)
        # d_sample = torch.from_numpy(d_sample).long().to(self.device)
        #
        # return s_sample, a_sample, r_sample, s_prime_sample, d_sample
        # ------------------------------------------------------------------------------
        # ------------------------------------------------------------------------------

        # experiences = random.sample(self.memory, k=batch_size)
        # states = torch.from_numpy(
        #     np.vstack([experience.state for experience in experiences if experience is not None])).float().to(self.device)
        # actions = torch.from_numpy(
        #     np.vstack([experience.action for experience in experiences if experience is not None])).long().to(self.device)
        # rewards = torch.from_numpy(
        #     np.vstack([experience.reward for experience in experiences if experience is not None])).float().to(self.device)
        # next_states = torch.from_numpy(
        #     np.vstack([experience.next_state for experience in experiences if experience is not None])).float().to(
        #     self.device)
        # # Convert done from boolean to int
        # dones = torch.from_numpy(
        #     np.vstack([experience.done for experience in experiences if experience is not None]).astype(
        #         np.uint8)).float().to(self.device)
        #
        # return states, actions, rewards, next_states, dones
