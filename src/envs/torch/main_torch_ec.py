'''
epoches = 1 # each sample is a full dataset. online learning
batch = 1 # online learning. each sample leads to weights change
'''
import numpy as np 
import cv2 
import matplotlib.pyplot as plt
import PIL.Image as Image
import gym
import random

from gym import Env, spaces
import time
from AI import AI
from WheeledRobots import WheeledRobots

from IPython import display


def plot_learning_curve(x, scores, figure_file):
    running_avg = np.zeros(len(scores))
    for i in range(len(running_avg)):
        running_avg[i] = np.mean(scores[max(0, i-100):(i+1)])
    plt.plot(x, running_avg)
    plt.title('Running average of previous 100 scores')
    plt.savefig(figure_file)


agents_amount = 2
env = WheeledRobots(agents_amount)


N = 20
batch_size = 5
n_epochs = 4
alpha = 0.0003
agents_ai = []
# print(env.get_action_dim()[0])
for i in range(agents_amount):
    agents_ai.append(AI(n_actions=env.get_action_dim()[0], batch_size=batch_size, 
                    alpha=alpha, n_epochs=n_epochs, 
                    input_dims=env.observation_space.shape))

n_games = 300

figure_file = 'plots/cartpole.png'

best_score = env.reward_range[0]
score_history = []

learn_iters = 0
avg_score = 0
n_steps = 0


loaded_ppo = False
for i in range(n_games):
    observation = env.reset()
    if not loaded_ppo:
        for a, ai in zip(env.world.agents, agents_ai):
            ai.load_models(a.name)
        loaded_ppo = True
    done = False
    score = 0
    while not done:
        # env.render()

        actions = []
        probs = []
        vals = []

        for ai in agents_ai:
            a, p, v = ai.choose_action(observation)
            actions.append(a)
            probs.append(p)
            vals.append(v)
            
        observation_, reward, done, info = env.step(actions)
        n_steps += 1
        score += reward
        
        for ai, a, p, v in zip(agents_ai, actions, probs, vals):
            ai.remember(observation, a, p, v, reward, done)
            if n_steps % N == 0:
                ai.learn()
                learn_iters += 1
        observation = observation_
    score_history.append(score)
    avg_score = np.mean(score_history[-100:])

    if avg_score > best_score:
        best_score = avg_score
        for ai, agent in zip(agents_ai, env.world.agents):
            ai.save_models(agent.name)

    print('episode', i, 'score %.1f' % score, 'avg score %.1f' % avg_score,
            'time_steps', n_steps, 'learning_steps', learn_iters)
x = [i+1 for i in range(len(score_history))]
plot_learning_curve(x, score_history, figure_file)

env.close()