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
from matplotlib import cm

def train(env, agent, max_episodes = 3000, max_steps = 2000):
    # print(max_steps)
    scores = []
    times = []
    # steps = []
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
            if mean >= 200:
                print("Solved")
                solved = True
                path = "solved_model.pth"
                torch.save(agent.qnet.state_dict(), path)
                break

        print("episode", n, "finished with reward", score, "in", step, "steps, with epsilon", agent.epsilon)
        if agent.epsilon > agent.min_epsilon:
            agent.epsilon *= agent.epsilon_decay
           # agent.epsilon = 1/(n+1)
        n += 1

    end = time.time()
    print("Trained for", end - start, "seconds")

    return scores, solved, times

def test(env, agent, max_episodes = 10):

    scores = []
    agent.epsilon = 0
    agent.min_epsilon = 0
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
        scores.append(score)

    return scores

def moving_average(scores, n=100) :
    ret = np.cumsum(scores, dtype=np.float64)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n-1:] / n

def plot_reward(scores, for_train = True):

    plt.figure(0)
    # plt.axis('scaled')
    if for_train:
        plt.title("Reward per training episode")
        plt.plot(scores, linewidth=0.25, color="b", label="Score")
        SMA100 = moving_average(scores, n=100)
        plt.plot(np.arange(len(scores)-len(SMA100), len(scores)), SMA100, label = "SMA100", color = "m")
    else:
        plt.title("Reward per testing episode")
        plt.plot(scores, linewidth=0.5, color="b", label="Score")
    plt.xlabel("n, Episode")
    plt.ylabel("Score")
    plt.axhline(y = 0, linewidth = 0.39, color = "k", linestyle = "--")
    plt.axhline(y = 200, linewidth = 0.65, color = "r", linestyle = "--")
    plt.grid(linewidth=0.2)
    plt.legend()
    if for_train:
        plt.savefig("reward_per_training_eps")
    else:
        plt.savefig("reward_per_testing_eps")
    plt.close()

def plot_scores_per_epsilon(score_per_epsilondecay, epsilon_decay_range):

    plt.figure(0)
    plt.title("50-episode Moving Average (with log x-axis)")
    colorindex = 0
    for i in range(len(epsilon_decay_range)):
        SMA50 = moving_average(score_per_epsilondecay[i], n=50)
        print("epsilon="+str(epsilon_decay_range[i])+" size:" + str(len(SMA50)))
        plt.plot(SMA50, label = "epsilon decay=" + str(epsilon_decay_range[i]), color = cm.gnuplot2(colorindex), linewidth = 0.7)
        colorindex += 1/len(epsilon_decay_range)
    plt.xlabel("Episodes")
    plt.ylabel("Score")
    plt.xscale("log")
    plt.axhline(y=0, linewidth=1, color="k", linestyle="--")
    plt.axhline(y=200, linewidth=0.5, color="k", linestyle="--")
    plt.grid(linewidth=0.2)
    plt.legend()
    plt.savefig("epsilon1")
    plt.close()

def plot_scores_per_epsilon_time(scores, times, epsilon_decay_range):

    assert len(scores) == len(times), "scores and times must have equal length"
    plt.figure(0)
    plt.title("100-episode Moving Average vs. Time (sec)")
    colorindex = 0
    for i in range(len(epsilon_decay_range)):
        SMA100 = moving_average(scores[i], n=100)
        time100 = moving_average(times[i], n=100)
        print("epsilon="+str(epsilon_decay_range[i])+" size:" + str(len(SMA100)))
        plt.plot(time100, SMA100, linewidth=0.7, color=cm.gnuplot2(colorindex), label="epsilon decay="+str(epsilon_decay_range[i]))
        colorindex += 1/len(epsilon_decay_range)
    plt.xlabel("Time (in seconds)")
    plt.ylabel("Score")
    plt.axhline(y = 0, linewidth = 1, color = "k", linestyle = "--")
    plt.axhline(y = 200, linewidth = 0.5, color = "k", linestyle = "--")
    plt.grid(linewidth=0.2)
    plt.legend()
    plt.savefig("epsilon2")
    plt.close()


def plot_scores_per_architecture(scores, architectures):

    plt.figure(0)
    plt.title("50-episode Moving Average (with log x-axis)")
    colorindex = 0
    for i in range(len(architectures)):
        SMA50 = moving_average(scores[i], n=50)
        # print(SMA50)
        plt.plot(SMA50, label = "Hidden Layer Sizes = " + str(architectures[i]), color = cm.jet(colorindex), linewidth = 0.6)
        colorindex += 1/len(architectures)
    plt.xlabel("Episodes")
    plt.ylabel("Score")
    plt.xscale("log")
    plt.axhline(y=0, linewidth=1, color="k", linestyle="--")
    plt.axhline(y=200, linewidth=0.5, color="k", linestyle="--")
    plt.grid(linewidth=0.2)
    plt.legend(fontsize="small")
    plt.savefig("architecture1")
    plt.close()


def plot_scores_per_architecture_time(scores, times, architectures):

    assert len(scores) == len(times), "scores and times must have equal length"
    plt.figure(0)
    plt.title("100-episode Moving Average vs. Time (sec)")
    colorindex = 0
    for i in range(len(architectures)):
        SMA100 = moving_average(scores[i], n=100)
        time100 = moving_average(times[i], n=100)
        # print(len(SMA25))
        plt.plot(time100, SMA100, linewidth=0.6, color=cm.jet(colorindex), label = "Hidden Layer Sizes = " + str(architectures[i]))
        colorindex += 1/len(architectures)
    plt.xlabel("Time (in seconds)")
    plt.ylabel("Score")
    plt.axhline(y = 0, linewidth = 1, color = "k", linestyle = "--")
    plt.axhline(y = 200, linewidth = 0.5, color = "k", linestyle = "--")
    plt.grid(linewidth=0.2)
    plt.legend(fontsize="small")
    plt.savefig("architecture_time1")
    plt.close()

def plot_scores_per_gamma(scores, gamma):

    plt.figure(5)
    plt.title("50-episode Moving Average (with log x-axis)")
    colorindex = 0
    for i in range(len(gamma)):
        SMA50 = moving_average(scores[i], n=50)
        # print(SMA50)
        plt.plot(SMA50, label = r"$\gamma = $" + str(gamma[i]), color = cm.jet(colorindex), linewidth = 0.6)
        colorindex += 1/len(gamma)
    plt.xlabel("Episodes")
    plt.ylabel("Score")
    plt.xscale("log")
    plt.axhline(y=0, linewidth=1, color="k", linestyle="--")
    plt.axhline(y=200, linewidth=0.5, color="k", linestyle="--")
    plt.grid(linewidth=0.2)
    plt.legend(fontsize="small")
    plt.savefig("gamma1")
    plt.close()


def plot_scores_per_gamma_time(scores, times, gamma):

    assert len(scores) == len(times), "scores and times must have equal length"
    plt.figure(6)
    plt.title("100-episode Moving Average vs. Time (sec)")
    colorindex = 0
    for i in range(len(gamma)):
        SMA100 = moving_average(scores[i], n=100)
        time100 = moving_average(times[i], n=100)
        # print(len(SMA25))
        plt.plot(time100, SMA100, linewidth=0.6, color=cm.jet(colorindex), label = r"$\gamma = $" + str(gamma[i]))
        colorindex += 1/len(gamma)
    plt.xlabel("Time (in seconds)")
    plt.ylabel("Score")
    plt.axhline(y = 0, linewidth = 1, color = "k", linestyle = "--")
    plt.axhline(y = 200, linewidth = 0.5, color = "k", linestyle = "--")
    plt.grid(linewidth=0.2)
    plt.legend(fontsize="small")
    plt.savefig("gamma_time1")
    plt.close()

def plot_scores_per_tuf(scores, target_update_freq):

    plt.figure(7)
    plt.title("50-episode Moving Average (log x-axis)")
    colorindex = 0
    for i in range(len(target_update_freq)):
        SMA50 = moving_average(scores[i], n=50)
        # print(SMA50)
        plt.plot(SMA50, label = "Freq. = " + str(target_update_freq[i]), color = cm.jet(colorindex), linewidth = 0.6)
        colorindex += 1/len(target_update_freq)
    plt.xlabel("Episodes")
    plt.ylabel("Score")
    plt.xscale("log")
    plt.axhline(y=0, linewidth=1, color="k", linestyle="--")
    plt.axhline(y=200, linewidth=0.5, color="k", linestyle="--")
    plt.grid(linewidth=0.2)
    plt.legend(fontsize="small")
    plt.savefig("tuf1")
    plt.close()


def plot_scores_per_tuf_time(scores, times, target_update_freq):

    assert len(scores) == len(times), "scores and times must have equal length"
    plt.figure(8)
    plt.title("100-episode Moving Average vs. Time (sec)")
    colorindex = 0
    for i in range(len(target_update_freq)):
        SMA100 = moving_average(scores[i], n=100)
        time100 = moving_average(times[i], n=100)
        # print(len(SMA25))
        plt.plot(time100, SMA100, linewidth=0.6, color=cm.jet(colorindex), label = "Freq = " + str(target_update_freq[i]))
        colorindex += 1/len(target_update_freq)
    plt.xlabel("Time (in seconds)")
    plt.ylabel("Score")
    plt.axhline(y = 0, linewidth = 1, color = "k", linestyle = "--")
    plt.axhline(y = 200, linewidth = 0.5, color = "k", linestyle = "--")
    plt.grid(linewidth=0.2)
    plt.legend(fontsize="small")
    plt.savefig("tuf_time1")
    plt.close()

def main():

    env = gym.make('LunarLander-v2')
    # # env = wrappers.Monitor(env, "./gym-results", force=True)

    # ------------------------------------------------------------------------------------------------------------
    # Plot training: pre-tuning
    # ------------------------------------------------------------------------------------------------------------

    agent = DQNAgent(batch_size = 128, buffer_size = 100000, epsilon_decay = 0.999,
                     gamma = 0.99, hidden_size1 = 128, hidden_size2 = 128, learning_rate = 0.0005,
                     regularization = 0.000001, momentum = 0.99, qnet_update_freq = 5)
    scores, solved, times = train(env, agent, max_episodes = 3000, max_steps = 2000)
    plot_reward(scores, for_train= True)

    # ------------------------------------------------------------------------------------------------------------
    # Plot training: post-tuning
    # ------------------------------------------------------------------------------------------------------------
    agent = DQNAgent(batch_size = 64, buffer_size = 100000, epsilon_decay = 0.99,
                     gamma = 0.99, hidden_size1 = 512, hidden_size2 = 512, learning_rate = 0.0005,
                     regularization = 0.000001, momentum = 0.99, qnet_update_freq = 5) #437.2386517524719 seconds
    scores, solved, times = train(env, agent, max_episodes = 3000, max_steps = 2000)
    plot_reward(scores, for_train= True)

    if solved:
        path = "solved_model.pth"
    else:
        path = "trained_model.pth"

    # ------------------------------------------------------------------------------------------------------------
    # Plot testing
    # ------------------------------------------------------------------------------------------------------------
    agent.qnet.load_state_dict(torch.load(path))
    scores = test(env, agent, max_episodes = 100)
    plot_reward(scores, for_train= False)

    #------------------------------------------------------------------------------------------------------------
    # Plot epsilon decay variation
    #------------------------------------------------------------------------------------------------------------

    epsilon_decay_range = [0.9999, 0.999, 0.99, 0.9]

    scores_per_epsilon = []
    times_per_epsilon = []

    for i in range(len(epsilon_decay_range)):
        agent = DQNAgent(batch_size = 64, buffer_size = 100000, epsilon_decay = epsilon_decay_range[i],
                         gamma = 0.99, hidden_size1 = 128, hidden_size2 = 128, learning_rate = 0.0005,
                         regularization = 0.000001, momentum = 0.99, qnet_update_freq = 5)
        scores, solved, times = train(env, agent, max_episodes = 3000, max_steps = 2000)
        scores_per_epsilon.append(scores)
        times_per_epsilon.append(times)

    np.save(arr=np.array(scores_per_epsilon, dtype = object), file="scores_per_epsilon")
    np.save(arr=np.array(times_per_epsilon, dtype = object), file="times_per_epsilon")
    scores_per_epsilon = np.load("scores_per_epsilon.npy", allow_pickle= True).tolist()
    times_per_epsilon = np.load("times_per_epsilon.npy", allow_pickle= True).tolist()

    plot_scores_per_epsilon(scores_per_epsilon, epsilon_decay_range)
    plot_scores_per_epsilon_time(scores_per_epsilon, times_per_epsilon, epsilon_decay_range)

    # ------------------------------------------------------------------------------------------------------------
    # Plot hidden layer size variation
    # ------------------------------------------------------------------------------------------------------------

    architecture = [(16,16), (32, 32), (64, 64), (128, 128), (256, 256), (512, 512)]
    # architecture = [(16, 32), (32, 64), (32, 128), (64, 128), (64, 256)]
    # architecture = [(256, 64), (128, 64), (128, 32), (64, 32), (64, 16), (32, 16)]
    scores_per_arch = []
    times_per_arch = []
    for i in range(len(architecture)):
        agent = DQNAgent(batch_size = 64, buffer_size = 100000, epsilon_decay = 0.99,
                         gamma = 0.99, hidden_size1 = architecture[i][0], hidden_size2 = architecture[i][1], learning_rate = 0.0005,
                         regularization = 0.000001, momentum = 0.99, qnet_update_freq = 5)
        scores, solved, times = train(env, agent, max_episodes = 3000, max_steps = 2000)
        scores_per_arch.append(scores)
        times_per_arch.append(times)
    np.save(arr=np.array(scores_per_arch, dtype = object), file="scores_per_arch")
    np.save(arr=np.array(times_per_arch, dtype = object), file="times_per_arch")

    scores_per_arch = np.load("scores_per_arch.npy", allow_pickle= True).tolist()
    times_per_arch = np.load("times_per_arch.npy", allow_pickle= True).tolist()

    plot_scores_per_architecture(scores_per_arch, architecture)
    plot_scores_per_architecture_time(scores_per_arch, times_per_arch, architecture)

    # ------------------------------------------------------------------------------------------------------------
    # Plot gamma variation
    # ------------------------------------------------------------------------------------------------------------

    gamma = [0.99, 0.95, 0.9, 0.8, 0.7, 0.5, 0.3, 0.1]
    scores_per_gamma = []
    times_per_gamma = []
    for i in range(len(gamma)):
        agent = DQNAgent(batch_size = 64, buffer_size = 100000, epsilon_decay = 0.99,
                         gamma = gamma[i], hidden_size1 = 512, hidden_size2 = 512, learning_rate = 0.0005,
                         regularization = 0.000001, momentum = 0.99, qnet_update_freq = 5)
        scores, solved, times = train(env, agent, max_episodes = 3000, max_steps = 2000)
        scores_per_gamma.append(scores)
        times_per_gamma.append(times)
    np.save(arr=np.array(scores_per_gamma, dtype = object), file="scores_per_gamma")
    np.save(arr=np.array(times_per_gamma, dtype = object), file="times_per_gamma")

    scores_per_gamma = np.load("scores_per_gamma.npy", allow_pickle= True).tolist()
    times_per_gamma = np.load("times_per_gamma.npy", allow_pickle= True).tolist()

    plot_scores_per_gamma(scores_per_gamma, gamma)
    plot_scores_per_gamma_time(scores_per_gamma, times_per_gamma, gamma)


    target_update_freq = [1, 3, 5, 7, 10, 30, 50, 100]
    scores_per_tuf = []
    times_per_tuf = []
    for i in range(len(target_update_freq)):
        agent = DQNAgent(batch_size = 64, buffer_size = 100000, epsilon_decay = 0.99,
                         gamma = 0.99, hidden_size1 = 512, hidden_size2 = 512, learning_rate = 0.0005,
                         regularization = 0.000001, momentum = 0.99, qnet_update_freq = 5)
        scores, solved, times = train(env, agent, max_episodes = 3000, max_steps = 1000)
        scores_per_tuf.append(scores)
        times_per_tuf.append(times)
    np.save(arr=np.array(scores_per_tuf, dtype = object), file="scores_per_tuf")
    np.save(arr=np.array(times_per_tuf, dtype = object), file="times_per_tuf")

    scores_per_tuf = np.load("scores_per_tuf.npy", allow_pickle= True).tolist()
    times_per_tuf = np.load("times_per_tuf.npy", allow_pickle= True).tolist()

    plot_scores_per_tuf(scores_per_tuf, target_update_freq)
    plot_scores_per_tuf_time(scores_per_tuf, times_per_tuf, target_update_freq)

    env.close()

if __name__ == '__main__':
    main()