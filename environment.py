import numpy as np
import gym

class LunarLander():

    def initialize(self, env_info={}):

        self.env = gym.make("LunarLander-v2")

    def start(self):

        reward = 0.0
        observation = self.env.reset()
        is_terminal = False
        self.reward_obs_term = (reward, observation, is_terminal)

        return self.reward_obs_term[1]

    def step(self, action):


        last_state = self.reward_obs_term[1]
        current_state, reward, is_terminal, _ = self.env.step(action)

        self.reward_obs_term = (reward, current_state, is_terminal)

        return self.reward_obs_term