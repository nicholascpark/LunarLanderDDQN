import numpy as np
import gym
from gym import wrappers
import torch
from agent import DQNAgent
import time
import matplotlib.pyplot as plt
import os
import yaml
import argparse

def train(env, agent, max_episodes = 3000, max_steps = 1000):

    scores = []
    times = []
    start = time.time()
    n = 0
    solved = False

    while n < max_episodes:
        s = env.reset()
        score = 0
        for step in range(max_steps):
            a = agent.pick_action(s)
            s_prime, r, done, _ = env.step(a)
            experience = s, a, r, s_prime, done
            agent.learn(experience)
            s = s_prime
            score += r
            if done:
                break

        scores.append(score)
        times.append(time.time() - start)

        if n >= 100:
            mean = np.mean(scores[-100:])

            # if eps % 5 == 0:
            #     print('Episode {}: 100SMA = {}'.format(eps, mean))
            #     path = "trained_model.pth"
            #     torch.save(agent.qnet.state_dict(), path)

            if mean >= 200:
                print("Solved")
                solved = True
                path = "solved_model.pth"
                torch.save(agent.qnet.state_dict(), path)
                break
        print("episode", n, "=", score, "epsilon", agent.epsilon)
        if agent.epsilon > 0.1:
            agent.epsilon *= agent.epsilon_decay
        n += 1

    end = time.time()
    print("Trained for", end - start, "seconds")

    return scores, solved, times

def moving_average(scores, n=100) :
    ret = np.cumsum(scores, dtype=np.float64)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n-1:] / n


def plot_reward(scores, times):

    plt.figure(0)
    plt.title("Reward per training episode")
    plt.plot(scores)
    SMA100 = moving_average(scores, n = 100)
    plt.plot(SMA100, label = "SMA100")
    plt.xlabel("n, Episode")
    plt.ylabel("Score")
    plt.grid(linewidth=0.2)
    plt.legend()
    plt.show()
    # plt.savefig("reward_per_training_eps")




def test(env, agent, max_episodes = 10):

    agent.epsilon = 0
    for i in range(max_episodes):
        score = 0
        s = env.reset()
        while True:
            action = agent.pick_action(s)
            s_prime, r, done, _ = env.step(action)
            s = s_prime
            score += r
            if done:
                break
        print('episode, {} scored {}'.format(i, score))

def main():

    env = gym.make('LunarLander-v2')
    # env = wrappers.Monitor(env, "./gym-results", force=True)
    agent = DQNAgent(batch_size = 64, buffer_size = 100000, epsilon_decay = 0.999,
                     gamma = 0.99, hidden_size1 = 32, hidden_size2 = 32, learning_rate = 0.0001,
                     regularization = 0.000001, momentum = 0.99, qnet_update_freq = 5)
    # solved = True
    scores, solved, times = train(env, agent, max_episodes = 3000, max_steps = 2000)
    plot_reward(scores, times)
    if solved:
        path = "solved_model.pth"
    else:
        path = "trained_model.pth"
    agent.qnet.load_state_dict(torch.load(path))
    test(env, agent, max_episodes = 10)
    env.close()

if __name__ == '__main__':
    main()