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
from ents.AI import AI
from ents.Networks import Networks
from ents.Agent import Agent
from ents.WheeledRobots import WheeledRobots
from ents.nn import Network

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



for i in range(n_games):
    observation = env.reset()
    done = False
    score = 0
    while not done:
        env.render()

        actions = []
        probs = []
        vals = []

        for ai in agents_ai:
            a, p, v = ai.choose_action(observation)
            actions.append(a)
            probs.append(p)
            vals.append(v)
        # action, prob, val = agent.choose_action(observation)
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
        for ai in agents_ai:
            ai.save_models()

    print('episode', i, 'score %.1f' % score, 'avg score %.1f' % avg_score,
            'time_steps', n_steps, 'learning_steps', learn_iters)
x = [i+1 for i in range(len(score_history))]
plot_learning_curve(x, score_history, figure_file)







# nepisodes = 3
# networks = Networks(env)


# popsize = 10
# pop_half = int(popsize / 2)
# variance = 0.1
# pertubation_variance = 0.02
# ngenerations = 500
# nepisodes = 200

# parameters = networks.compute_nparameters()
# populations = []
# for param in parameters:
#     populations.append(np.random.randn(popsize, param) * variance)
# new_pops = np.transpose(populations, (1, 0, 2))



# # print(np.array(populations).shape)
# for g in range(ngenerations):
#     print("____________Generation ", g, " :\n")
#     fitness = []
#     for i in range(popsize):
#         # print(i)
#         networks.set_genotypes(new_pops[i])
#         fit = networks.evaluate(env, nepisodes, show = True)
#         fitness.append(fit)
# #    print(fitness)
#     best_index = np.argsort(fitness)
#     print(" Best fitness: ", best_index[0])
#     # print(best_index)
#     # print(new_pops.shape)
#     # print(np.array(parameters).shape)
# #    print("\n best_index ", best_index)
#     for i in range(pop_half):
#         for a in range(agents_amount):
#             new_pops[best_index[i]] = new_pops[best_index[i+pop_half]] + np.random.randn(parameters[a]) * pertubation_variance
#     print( " - total rewards: ", fitness, "\n____________Generation ", g, ".")


    

















env.close()
'''
for i_episode in range(10):

    observation = env.reset()
    fitness = 0

    for t in range(200):
        #env.render()

        env_screen = env.render()

        #print(observation)
        action = env.action_space.sample()
        # print("\n___"+str(action)+"___\n")
        observation, reward, done, info = env.step(action)
        fitness += reward
        # print("\nFitness: "+str(fitness)+"\n")

        if done:
            print("Episode finished after {} timesteps".format(t+1))
            print("\nFitness: "+str(fitness)+"\n")

            break
env.close()

while True:
    # Take a random action
    action = env.action_space.sample()
    obs, reward, done, info = env.step(action)
    
    # Render the game
    env.render()
    
    if done == True:
        break

env.close()
'''