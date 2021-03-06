import numpy as np
import random
from Element import Element
from Point import Point
from Agent import Agent
import cv2
import PIL.Image as Image
from time import sleep

from Bird import Bird

font = cv2.FONT_HERSHEY_COMPLEX_SMALL 
class World:
    def __init__(self, form, agents_amount, objects_amount, show = True):
        self.agents = []
        self.objects = []
        self.elements = []
        self.agents_amount = agents_amount
        self.objects_amount = objects_amount
        self.canvas_shape = form

        # print(self.canvas_shape)

        # Init the canvas 
        self.canvas = np.ones(form)

        # print(self.canvas_shape)
        

        # Permissible area of helicper to be 
        self.y_min = int (form[1] * 0.1)
        self.x_min = int (form[0] * 0.1)
        self.y_max = int (form[1] * 0.9)
        self.x_max = int (form[0] * 0.9)
        # print(self.x_min, self.y_min, self.x_max, self.y_max)
        for i in range(self.agents_amount):

            agent = Agent("Agent_{}".format(i), self.x_max, self.x_min, self.y_max, self.y_min, show)
            agent.set_active_collision(True)
            self.agents.append(agent)
            self.elements.append(agent)

    def assign_ai(self, ais):
        for a, ai in zip(self.agents, ais):
            a.ai = ai
    
    def get_agents(self):
        return self.agents
    
    def set_objects(self, objects):
        self.objects = objects
        self.elements.extend(objects)

    def get_objects(self):
        return self.objects

    def has_collided(self, elem1, elem2):

        if not elem1.collided() or not elem2.collided() or elem1.name == elem2.name:
            return False

        elem1_x, elem1_y = elem1.get_position()
        elem2_x, elem2_y = elem2.get_position()

        if (2 * abs(elem1_x - elem2_x) <= (elem1.get_w() + elem2.get_w())) \
            and \
            (2 * abs(elem1_y - elem2_y) <= (elem1.get_h() + elem2.get_h())):
            return True

        return False

    def reset_agents(self):
        
        # надо новых агентов создать сохранив их разумы
        for a in self.agents:
            a.agent_reset()

            drawn = False
            while not drawn:
                x = np.float16(random.randrange(self.x_min, self.x_max))
                y = np.float16(random.randrange(self.y_min, self.y_max))
                a.set_position(x, y)
                if self.agents_amount == 1:
                    self.elements.append(a)
                    drawn = True
                for other in self.agents:
                    if not a.name == other.name:
                        if not self.has_collided(a, other):
                            self.elements.append(a)
                            drawn = True
                        else:
                            x = np.float16(random.randrange(self.x_min, self.x_max))
                            y = np.float16(random.randrange(self.y_min, self.y_max))
                            a.set_position(x, y)

        
    def reset(self):
        
        self.objects = []
        self.elements = []
        self.reset_agents()
        # Determine a place to intialise the chopper in
        # print("here")
        
        for i in range(self.objects_amount):

            x = np.float16(random.randrange(self.x_min, self.x_max))
            y = np.float16(random.randrange(self.y_min, self.y_max))

            object = Bird("Bird_{}".format(i), self.x_max, self.x_min, self.y_max, self.y_min)
            object.set_position(x, y)
            self.objects.append(object)
            self.elements.append(object)


    def draw_cross(self, canvas, x, y):
        canvas[x-5 : x+5, y-5 : y+5] = 0
        return canvas
    
    
    def draw_world(self, left, reward, done = False, dark = False):
        # print("-------------------------")
        self.canvas[self.x_min, :] = 0
        self.canvas[self.x_max, :] = 0
        self.canvas[:, self.y_min] = 0
        self.canvas[:, self.y_max] = 0

        if dark:
            self.canvas = np.zeros(self.canvas_shape)
        else:
            self.canvas = np.ones(self.canvas_shape)
        # print(self.elements)
        if not done:
            # Draw the heliopter on canvas
            for elem in self.elements:
                elem_shape = (elem.get_w(), elem.get_h())
                x, y = elem.x, elem.y
                # print(x,y)
                # print(np.round(np.float32(x)))
                # print(elem.name)
                self.canvas[int(np.round(np.float16(x)) - elem_shape[0]/2) : int(np.round(np.float16(x)) + elem_shape[0]/2), int(np.round(np.float16(y)) - elem_shape[0]/2): int(np.round(np.float16(y)) + elem_shape[1]/2)] = elem.icon
                # self.canvas = self.draw_cross(self.canvas, int(np.round(x)), int(np.round(y)))

                # self.canvas = cv2.putText(self.canvas, elem.name, (int(y - 20), int(x)), font,  
                # 0.8, (0,0,0), 1, cv2.LINE_AA)

        text = 'Timer: {} | Rewards: {}'.format(left, reward)

        # Put the info on canvas 
        self.canvas = cv2.putText(self.canvas, text, (10,20), font,  
                0.8, (0,0,0), 1, cv2.LINE_AA)
        return self.canvas
    
    def get_world(self):
        return self.canvas
    
    def get_observation(self):
        obs = []
        for a in self.agents:
            x,y = a.get_position()
            obs.append(x / self.canvas_shape[0])
            obs.append(y / self.canvas_shape[1])
            obs.append(a.orientation / 360.0)
            obs.append(a.velocity)
        for o in self.objects:
            x,y = o.get_position()
            obs.append(x / self.canvas_shape[0])
            obs.append(y / self.canvas_shape[1])
        return obs

    def get_com_observation(self):
        obs = []
        for a in self.agents:
            for act in a.message:
                obs.append(act)
        return obs
    