

from time import sleep
import numpy as np 
import cv2
import random
import matplotlib.pyplot as plt
import PIL.Image as Image
from gym import Env, spaces
from Agent import Agent
from World import World
from math import sin, cos, sqrt, radians

class WheeledRobots(Env):
    def __init__(self, agents_amount, show = True):
        self.reward = 0
        self.show = show
        self.episodes_number = 100
        self.min_turn = 0.0
        self.max_turn = 0.0
        self.min_velo = 0.0
        self.max_velo = 0.0
        self.min_signal = 0.0
        self.max_signal = 0.0
        self.overall_distance = 0.0
        self.agents_amount = agents_amount
        super(WheeledRobots, self).__init__()
        
        # Define a 2-D observation space
        # observation_shape = (600, 800, 3)
        world_shape = (600, 800, 3)
        obs_vector = 0
        for i in range(agents_amount):
            # velocity, orientation, position(x,y), birds position(x,y)
            obs_vector += 8

        # self.observation_space = spaces.Box(low = np.zeros(observation_shape), 
        #                                 high = np.ones(observation_shape),
        #                                 dtype = np.float16)
    
        self.observation_space = spaces.Box(low = np.zeros(obs_vector), 
                                        high = np.ones(obs_vector),
                                        dtype = np.float16)
    
        
        # Define an action space ranging from 0 to 4
        # self.action_space = spaces.Discrete(5,)
        self.action_spaces = []
        for i in range(self.agents_amount):
            self.action_spaces.append(spaces.Box(low= np.zeros(8),
                                        high= np.ones(8),
                                        dtype=np.float16))
        
        self.world = World(world_shape, self.agents_amount, self.agents_amount, self.show)
        
        self.finish_distance = 0.0
        # Define elements present inside the environment
        
        # Maximum fuel chopper can take at once
        # self.max_fuel = 1000
    

    def get_agents_amount(self):
        return self.agents_amount
    
    # def reward_goal_1(self, actions):
    #     agents = self.world.agents
    #     if agents[0].velocity < actions[0][1] and agents[1].velocity > actions[1][1]:
    #         return 1
    #     else:
    #         return 0
    def reward_goal_1(self):
        agents = self.world.agents
        rew = 0
        # for b in self.world.objects:
        #     print(b.name)
        for agent in self.world.agents:
            for b in  self.world.objects:
                # print(b.name)
                if self.world.has_collided(agent, b):
                    # print("Bird collision")
                    rew += 1

        return rew
            # print(agent.get_distance())
            # # print(agent.get_distance_old())
            # if agent.get_distance() < agent.get_distance_old():

            #     agent.distance_old = agent.distance
            #     return 1
            # else:
            #     agent.distance_old = agent.distance
            #     return -1
            
                # if agents[0].velocity < actions[0][1] and agents[1].velocity > actions[1][1]:
                #     return 1
                # else:
                #     return 0

    def reset(self):
        # Reset the fuel consumed
        # self.fuel_left = self.max_fuel

        # Reset the reward
        self.ep_return  = self.episodes_number
        # Number of birds
        # self.bird_count = 0
        # self.fuel_count = 0
        # self.agent_count = 0

        # print("-----------here")
        self.world.reset()

        # self.finish_distance = 0.0
        # for i in range(self.world.agents_amount):
        #     self.finish_distance += self.world.agents[i].icon_w_original / 2 + 20

        # Determine a place to intialise the chopper in
        
        # Intialise the chopper

        # Intialise the elements 

        # Reset the Canvas 
        # self.canvas = np.ones(self.observation_shape) * 1

        # Draw elements on the canvas
        # self.observation_space = self.world.draw_world()
        self.observation_space = self.world.get_observation()

        # return the observation
        return self.observation_space
    
    def get_action_meanings(self):
        #return {0: "Right", 1: "Left", 2: "Down", 3: "Up", 4: "Do Nothing"}
        return {0: "Speed", 1: "Turn", 2: "Signal"}

    def get_action_dim(self):
        return self.action_spaces[0].shape
    

    

    def step(self, actions):
        # Flag that marks the termination of an episode
        done = False
        
        # Assert that it is a valid action 
        # assert self.action_space.contains(action), "Invalid Action"
        
        # assert (len(action) == 3), "Invalid Action"
        # print(actions)

        # Decrease the fuel counter 
        # self.fuel_left -= 1 
        
        # Reward for executing a step.

        self.reward = self.reward_goal_1()
        # reward = 1

        # current_distance = 0.0
        # for i in range(self.world.agents_amount-1):
        #     x1, y1 = self.world.agents[i].get_position()
        #     x2, y2 = self.world.agents[i-1].get_position()
        #     current_distance += sqrt((x1 - x2)**2 + (y1 - y2)**2)

        # if current_distance <= self.finish_distance:
        #     done = True
        #     reward = self.ep_return

        # if current_distance < self.overall_distance:
        #     reward = 1
        # else:
        #     reward = 0

        # self.overall_distance = current_distance

        # print("current: ", current_distance, "finish: ", self.finish_distance)

            # print("current: ", current_distance, "finish: ", self.finish_distance, "------------------done condition")
        
                # Increment the episodic return
        self.ep_return -= 1


        if(self.ep_return <= 0):
            done = True

        # apply the action to the chopper
        # print("Action turn : ", action[1])
        # print("Action velo : ", action[0])
        for agent, action in zip(self.world.agents, actions):
            vel = agent.get_agent_velocities(action)
            # print(actions)
            # if self.min_signal > action[2]:
            #     self.min_signal  = action[2]
            # if self.max_signal < action[2]:
            #     self.max_signal  = action[2]
            # if self.min_turn > action[0]:
            #     self.min_turn  = action[0]
            # if self.max_signal < action[0]:
            #     self.max_signal  = action[0]
            # if self.min_velo > action[1]:
            #     self.min_velo  = action[1]
            # if self.max_velo < action[1]:
            #     self.max_velo  = action[1]
            
            # print("\n", action.item(), "\n")
            # print(action)
            # print(vel[5])
            agent.rotate(vel[5])
            agent.set_velocity(sqrt(vel[1]**2 + vel[0]**2))
            # agent.set_signal(action[2])
            # agent.set_velocity(action*10)

            current_x, current_y = agent.get_position()
            cur_x = float(current_x)
            cur_y = float(current_y)
            # print("Velocity: ", agent.velocity, "\n====================\ncurrent_x_y:", current_x, current_y)
            agent.move_agent()

            # print("after_move. current_x_y \n", current_x, current_y)
            # print("after_move. cur_x_y \n", cur_x, cur_y)
            # f = agent.get_position()
            # print(f[0], f[1])

            # see next move is collision with 
            # other agents, if yes, than set prev location
            if any(self.world.has_collided(agent, a) for a in self.world.agents):
                # print("Touch an agent")

                agent.set_position(cur_x, cur_y)

        self.observation_space = self.world.get_observation()
        # print(self.world.get_world().shape)
        # print(self.obsservation_space)
        return self.observation_space, self.reward, done, []

    def render(self, mode = "human"):
        if self.show:
            assert mode in ["human", "rgb_array"], "Invalid mode, must be either \"human\" or \"rgb_array\""
            if mode == "human":
                cv2.imshow("Game", self.world.draw_world(self.ep_return, self.reward))
                # print(self.world.get_world().shape)
                cv2.waitKey(10)
            
            elif mode == "rgb_array":
                return self.world.draw_world()
        
    def close(self):
        cv2.destroyAllWindows()
