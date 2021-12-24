import numpy as np
import torch

class ReplayMemory:

    def __init__(self, buffer_size = 200000):

        self.memory =[]
        self.buffer_size = buffer_size
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

