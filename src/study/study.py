import gym
import numpy as np

class Network:
    def __init__(self, env, pvariance = 0.1, nhiddens = 5):
        self.pvariance = pvariance
        self.nhiddens = nhiddens
        self.ninputs = env.observation_space.shape[0]

        if (isinstance(env.action_space, gym.spaces.box.Box)):
            self.noutputs = env.action_space.shape[0]
        else:
            self.noutputs = env.action_space.n

        self.w1 = np.zeros(shape = (self.nhiddens, self.ninputs))
        self.w2 = np.zeros(shape = (self.noutputs, self.nhiddens))
        self.b1 = np.zeros(shape = (self.nhiddens, 1))
        self.b2 = np.zeros(shape = (self.noutputs, 1))


    def initparameters(self):
        self.w1 = np.random.randn(self.nhiddens, self.ninputs) * self.pvariance
        self.w2 = np.random.randn(self.noutputs, self.nhiddens) * self.pvariance
        self.b1 = np.zeros(shape=(self.nhiddens, 1))
        self.b2 = np.zeros(shape=(self.noutputs, 1))

    def compute_nparameters(self):
        nparameters = self.nhiddens * self.ninputs + self.noutputs * self.nhiddens + self.nhiddens + self.noutputs
        return nparameters

    def set_genotype(self, genotype):
        i1 = self.w1.size
        i2 = i1 + self.w2.size
        i3 = self.b1.size + i2

        self.w1 = np.reshape(genotype[0 : i1], (self.nhiddens, self.ninputs))
        self.w2 = np.reshape(genotype[i1 : i2], (self.noutputs, self.nhiddens))

        self.b1 = genotype[i2 : i3]
        self.b2 = genotype[i3 : ]

    def update(self, observation):
        observation.resize(self.ninputs, 1)
        z1 = np.dot(self.w1, observation) + np.reshape(self.b1, (self.b1.size, 1))
        a1 = np.tanh(z1)
        z2 = np.dot(self.w2, a1) + np.reshape(self.b2, (self.b2.size, 1))
        a2 = np.tanh(z2)

        if(isinstance(env.action_space, gym.spaces.box.Box)):
            action = a2
        else:
            action = np.argmax(a2)
        return  int(action)

    def evaluate(self, env, nepisodes, show = False):
        fitness = 0
        observation = env.reset()
        for i_episode in range(nepisodes):
            for t in range(100):
                if (show):
                    env.render()
                action = self.update(observation)
                observation, reward, done, info = env.step(action)

                fitness += reward
                if done:
                    #print("Episode finished after {} timesteps".format(t+1))
                    break


        return fitness / nepisodes



env = gym.make("CartPole-v0")
nepisodes = 3
network = Network(env)


popsize = 10
pop_half = int(popsize / 2)
variance = 0.1
pertubation_variance = 0.02
ngenerations = 100
nepisodes = 3

nparameters = network.compute_nparameters()
population = np.random.randn(popsize, nparameters) * variance


'''
#train
'''
for g in range(ngenerations):
    fitness = []
    for i in range(popsize):
        network.set_genotype(population[i])
        fit = network.evaluate(env, nepisodes)
        fitness.append(fit)
#    print(fitness)
    best_index = np.argsort(fitness)
#    print("\n best_index ", best_index)
    for i in range(pop_half):
        population[best_index[i]] = population[best_index[i+pop_half]] + np.random.randn(nparameters) * pertubation_variance

'''
#test
'''
for t in range(3):
    network.set_genotype(population[best_index[-1]])
    fitness = network.evaluate(env, nepisodes, show = True)
    print("\n Best fitness: ", str(fitness))
env.close()
'''

Acrobot-v1


env = gym.make("Acrobot-v1")
nepisodes = 3
network = Network(env)


popsize = 20
pop_half = int(popsize / 2)
variance = 0.1
pertubation_variance = 0.02
ngenerations = 100
nepisodes = 20

nparameters = network.compute_nparameters()
population = np.random.randn(popsize, nparameters) * variance



train

for g in range(ngenerations):
    fitness = []
    for i in range(popsize):
        network.set_genotype(population[i])
        fit = network.evaluate(env, nepisodes)
        fitness.append(fit)
#    print(fitness)
    best_index = np.argsort(fitness)
#    print("\n best_index ", best_index)
    for i in range(pop_half):
        population[best_index[i]] = population[best_index[i+pop_half]] + np.random.randn(nparameters) * pertubation_variance
    print("\nGeneration ", g, "\n")


test

for t in range(3):
    network.set_genotype(population[best_index[-1]])
    fitness = network.evaluate(env, nepisodes, show = True)
    print("\n Best fitness: ", str(fitness))
env.close()
'''
