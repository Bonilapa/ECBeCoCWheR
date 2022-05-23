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
from ents.Networks import Networks

from ents.ChopperScape import ChopperScape
from ents.nn import Network

from IPython import display

agents_amount = 5
env = ChopperScape(agents_amount)
nepisodes = 3
networks = Networks(env)


popsize = 10
pop_half = int(popsize / 2)
variance = 0.1
pertubation_variance = 0.02
ngenerations = 500
nepisodes = 200

parameters = networks.compute_nparameters()
populations = []
for param in parameters:
    populations.append(np.random.randn(popsize, param) * variance)
new_pops = np.transpose(populations, (1, 0, 2))



# print(np.array(populations).shape)
for g in range(ngenerations):
    print("____________Generation ", g, " :\n")
    fitness = []
    for i in range(popsize):
        # print(i)
        networks.set_genotypes(new_pops[i])
        fit = networks.evaluate(env, nepisodes, show = True)
        fitness.append(fit)
#    print(fitness)
    best_index = np.argsort(fitness)
    print(" Best fitness: ", best_index[0])
    # print(best_index)
    # print(new_pops.shape)
    # print(np.array(parameters).shape)
#    print("\n best_index ", best_index)
    for i in range(pop_half):
        for a in range(agents_amount):
            new_pops[best_index[i]] = new_pops[best_index[i+pop_half]] + np.random.randn(parameters[a]) * pertubation_variance
    print( " - total rewards: ", fitness, "\n____________Generation ", g, ".")

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