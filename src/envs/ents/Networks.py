
from ents.nn import Network
import numpy as np

class Networks:
    def __init__(self, env):
        self.networks = []
        for i in range(0,env.get_agents_amount()):
            self.networks.append(Network(env))
    

    def compute_nparameters(self):
        parameters = []
        for network in self.networks:
            parameters.append(network.compute_nparameters())
        return parameters
    

    def set_genotypes(self, genotypes):
        # print(np.array(genotypes).shape)
        for (genotype, network) in zip(genotypes, self.networks):
            # print(np.array(genotype).shape)
            network.set_genotype(genotype)

    def evaluate(self, env, nepisodes, show = False):
        fitness = 0
        observation = env.reset()
        for i_episode in range(nepisodes):
            for t in range(5):
                if (show):
                    env.render()
                actions = []

                for n in self.networks:
                    actions.append(n.update(observation))
                # print(action)
                # print("here", observation.shape)
                observation, reward, done, info = env.step(actions)

                fitness += reward
                if done:
                    print("Episode finished after {} timesteps".format(t+1))
                    break


        return fitness / nepisodes