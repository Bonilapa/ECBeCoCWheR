import numpy as np
import random
from ents.Element import Element
from ents.Point import Point
from ents.Agent import Agent
import cv2
import PIL.Image as Image
from time import sleep

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

    def set_agents(self, agents):
        self.agents = agents
        self.elements.extend(self.agents)
    
    def get_agents(self):
        return self.agents
    
    def set_objects(self, objects):
        self.objects = objects
        self.elements.extend(objects)

    def get_objects(self):
        return self.objects

    def has_collided(self, elem1, elem2):

        if not elem1.collided() or not elem2.collided():
            return False

        elem1_x, elem1_y = elem1.get_position()
        elem2_x, elem2_y = elem2.get_position()

        if (2 * abs(elem1_x - elem2_x) <= (elem1.get_w() + elem2.get_w())) \
            and \
            (2 * abs(elem1_y - elem2_y) <= (elem1.get_h() + elem2.get_h())):
            return True

        return False

    def reset(self):

        self.agents = []
        self.objects = []
        self.elements = []
        # Determine a place to intialise the chopper in
        # print("here")
        for i in range(self.agents_amount):

            agent = Agent("Agent_{}".format(i), self.x_max, self.x_min, self.y_max, self.y_min)
            agent.activate_collision()

            drawn = False

            while not drawn:
                x = np.float16(random.randrange(self.x_min, self.x_max))
                y = np.float16(random.randrange(self.y_min, self.y_max))
                agent.set_position(x, y)
                # print(agent.get_position())
                # sleep(1)
                # for a in self.agents:
                #     if self.has_collided(agent, a):
                #         print("True")
                #     else:
                #         print("False")
                # print(len(self.agents))
                if not any(self.has_collided(agent, a) for a in self.agents):
                    self.agents.append(agent)
                    self.elements.append(agent)
                    drawn = True
                # for i in range(len(self.agents)):
                #     if self.has_collided(agent, self.agents[i]):
                #         break
                #     else: 
                #         if i >= len(self.agents)-1:
                #             drawn = True

        
        for i in range(self.objects_amount):

            x = np.float16(random.randrange(self.x_min, self.x_max))
            y = np.float16(random.randrange(self.y_min, self.y_max))

            object = Element("Element_{}".format(i), self.x_max, self.x_min, self.y_max, self.y_min)
            object.set_position(x, y)
            self.objects.append(object)
            self.elements.append(object)


    def draw_cross(self, canvas, x, y):
        canvas[x-5 : x+5, y-5 : y+5] = 0
        return canvas
    
    
    def draw_world(self, done = False, dark = False):

        # print(self.canvas_shape)
        # print(self.canvas.shape)
        self.canvas[self.x_min, :] = 0
        self.canvas[self.x_max, :] = 0
        self.canvas[:, self.y_min] = 0
        self.canvas[:, self.y_max] = 0

        if dark:
            self.canvas = np.zeros(self.canvas_shape)
        else:
            self.canvas = np.ones(self.canvas_shape)

        if not done:
            # Draw the heliopter on canvas
            for elem in self.elements:
                elem_shape = (elem.get_w(), elem.get_h())
                # print("\n", elem_shape, "\n")
                # print(np.array((elem.icon)).shape)
                x, y = elem.x, elem.y
                # print(x,y)
                # print("[ ", elem.name, elem.get_position()) 
                # print(elem_shape, elem_shape[1], elem_shape[0])
                # temp = self.canvas[x:x + elem_shape[0], y : (y + elem_shape[1])]
                # print(temp.shape, self.canvas.shape[0], y, y + elem_shape[1], "\n")
                # print(elem.name)
                self.canvas[int(np.round(x) - elem_shape[0]/2) : int(np.round(x) + elem_shape[0]/2), int(np.round(y) -  + elem_shape[0]/2): int(np.round(y) + elem_shape[1]/2)] = elem.icon
                self.canvas = self.draw_cross(self.canvas, int(np.round(x)), int(np.round(y)))

                # self.canvas = cv2.putText(self.canvas, elem.name, (int(y - 20), int(x)), font,  
                # 0.8, (0,0,0), 1, cv2.LINE_AA)

        # text = 'Fuel Left: {} | Rewards: {}'.format(0, 0)

        # Put the info on canvas 
        # self.canvas = cv2.putText(self.canvas, text, (10,20), font,  
        #         0.8, (0,0,0), 1, cv2.LINE_AA)
        return self.canvas
    
    def get_world(self):
        return self.canvas
    