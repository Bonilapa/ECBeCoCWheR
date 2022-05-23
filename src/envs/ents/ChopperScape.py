

from time import sleep
import numpy as np 
import cv2
import random
import matplotlib.pyplot as plt
import PIL.Image as Image
from gym import Env, spaces
from ents.Agent import Agent
from ents.Bird import Bird
from ents.Fuel import Fuel
from ents.World import World
from math import sin, cos, sqrt, radians

class ChopperScape(Env):
    def __init__(self, agents_amount, show = True):
        self.overall_distance = 0.0
        self.agents_amount = agents_amount
        super(ChopperScape, self).__init__()
        
        # Define a 2-D observation space
        observation_shape = (600, 800, 3)
        self.observation_space = spaces.Box(low = np.zeros(observation_shape), 
                                        high = np.ones(observation_shape),
                                        dtype = np.float16)
    
        
        # Define an action space ranging from 0 to 4
        # self.action_space = spaces.Discrete(5,)
        self.action_spaces = []
        for i in range(0, self.agents_amount):
            self.action_spaces.append(spaces.Box(low= - np.ones(3),
                                        high= np.ones(3),
                                        dtype=np.float16))
        
        self.world = World(observation_shape, self.agents_amount, 0)
        
        # Define elements present inside the environment
        
        # Maximum fuel chopper can take at once
        # self.max_fuel = 1000
    

    def get_agents_amount(self):
        return self.agents_amount

    def reset(self):
        # Reset the fuel consumed
        # self.fuel_left = self.max_fuel

        # Reset the reward
        self.ep_return  = 0
        # Number of birds
        # self.bird_count = 0
        # self.fuel_count = 0
        # self.agent_count = 0

        # print("-----------here")
        self.world.reset()

        # Determine a place to intialise the chopper in
        
        # Intialise the chopper

        # Intialise the elements 

        # Reset the Canvas 
        # self.canvas = np.ones(self.observation_shape) * 1

        # Draw elements on the canvas
        self.observation_space = self.world.draw_world()

        # return the observation
        return self.observation_space
    
    def get_action_meanings(self):
        #return {0: "Right", 1: "Left", 2: "Down", 3: "Up", 4: "Do Nothing"}
        return {0: "Speed", 1: "Turn", 2: "Signal"}
    

    

    def step(self, actions):
        # Flag that marks the termination of an episode
        done = False
        
        # Assert that it is a valid action 
        # assert self.action_space.contains(action), "Invalid Action"
        
        # assert (len(action) == 3), "Invalid Action"
        # print(action)

        # Decrease the fuel counter 
        # self.fuel_left -= 1 
        
        # Reward for executing a step.
        reward = 0
        current_distance = 0.0
        for i in range(self.world.agents_amount-1):
            x1, y1 = self.world.agents[i].get_position()
            x2, y2 = self.world.agents[i-1].get_position()
            current_distance += sqrt((x1 - x2)**2 + (y1 - y2)**2)
        if current_distance < self.overall_distance:
            reward = 1
        else:
            reward = 0
        
        self.overall_distance = current_distance

        # apply the action to the chopper
        # print("Action turn : ", action[1])
        # print("Action velo : ", action[0])
        for agent, action in zip(self.world.agents, actions):
            agent.rotate(action[1]* 10)
            agent.set_velocity(action[0]*10)

            current_x, current_y = agent.get_position()
            cur_x = float(current_x)
            cur_y = float(current_y)
            # print("====================\ncurrent_x_y:", current_x, current_y)
            agent.move_agent()

            # print("after_move. current_x_y \n", current_x, current_y)
            # print("after_move. cur_x_y \n", cur_x, cur_y)
            # f = agent.get_position()
            # print(f[0], f[1])

            # see next move is collision with 
            # other agents, if yes, than set prev location
            if any( not (a == agent) and self.world.has_collided(agent, a) for a in self.world.agents):
                # print("Collision!!!!!!!!!!!!!!")

                agent.set_position(cur_x, cur_y)

            # f = agent.get_position()
            # print(f[0], f[1], "\n==============")
            
        # sleep(5)
        

        # # Spawn a bird at the right edge with prob 0.01
        # if random.random() < 0.01:
            
        #     # Spawn a bird
        #     spawned_bird = Bird("bird_{}".format(self.bird_count), self.x_max, self.x_min, self.y_max, self.y_min)
        #     self.bird_count += 1

        #     # Compute the x,y co-ordinates of the position from where the bird has to be spawned
        #     # Horizontally, the position is on the right edge and vertically, the height is randomly 
        #     # sampled from the set of permissible values
        #     bird_x = self.x_max
        #     bird_y = random.randrange(self.y_min, self.y_max)
        #     spawned_bird.set_position(self.x_max, bird_y)
            
        #     # Append the spawned bird to the elements currently present in Env. 
        #     self.elements.append(spawned_bird)    

        # # Spawn a fuel at the bottom edge with prob 0.01
        # if random.random() < 0.01:
        #     # Spawn a fuel tank
        #     spawned_fuel = Fuel("fuel_{}".format(self.fuel_count), self.x_max, self.x_min, self.y_max, self.y_min)
        #     self.fuel_count += 1
            
        #     # Compute the x,y co-ordinates of the position from where the fuel tank has to be spawned
        #     # Horizontally, the position is randomly chosen from the list of permissible values and 
        #     # vertically, the position is on the bottom edge
        #     fuel_x = random.randrange(self.x_min, self.x_max)
        #     fuel_y = self.y_max
        #     spawned_fuel.set_position(fuel_x, fuel_y)
            
        #     # Append the spawned fuel tank to the elemetns currently present in the Env.
        #     self.elements.append(spawned_fuel)   

        # For elements in the Ev
        # for elem in self.world.elements:

            # if isinstance(elem, Bird):
            #     # If the bird has reached the left edge, remove it from the Env
            #     if elem.get_position()[0] <= self.x_min:
            #         self.elements.remove(elem)
            #     else:
            #         # Move the bird left by 5 pts.
            #         elem.move(-5,0)
                
            #     # If the bird has collided.
            #     if self.has_collided(self.chopper, elem):
            #         # Conclude the episode and remove the chopper from the Env.
            #         reward = -10
            #         # self.elements.remove(self.chopper)
            #         done = True

            # if isinstance(elem, Fuel):
            #     # If the fuel tank has reached the top, remove it from the Env
            #     if elem.get_position()[1] <= float(self.y_min):
            #         self.elements.remove(elem)
            #     else:
            #         # Move the Tank up by 5 pts.
            #         elem.move(0, -5)
                    
            #     # If the fuel tank has collided with the chopper.
            #     if self.has_collided(self.chopper, elem):
            #         # Remove the fuel tank from the env.
            #         self.elements.remove(elem)
                    
            #         # Fill the fuel tank of the chopper to full.
            #         self.fuel_left = self.max_fuel
        
        # Increment the episodic return
        self.ep_return += 1

        # If out of fuel, end the episode.
        # if self.fuel_left <= 0:
        #     done = True

        # Draw elements on the canvas
        self.observation_space = self.world.draw_world(done = done)
        # print(self.world.get_world().shape)
        return self.observation_space, reward, done, []

    def render(self, mode = "human"):
        assert mode in ["human", "rgb_array"], "Invalid mode, must be either \"human\" or \"rgb_array\""
        if mode == "human":
            cv2.imshow("Game", self.world.draw_world())
            # print(self.world.get_world().shape)
            cv2.waitKey(10)
        
        elif mode == "rgb_array":
            return self.world.draw_world()
        
    def close(self):
        cv2.destroyAllWindows()
