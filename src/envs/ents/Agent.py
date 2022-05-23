from xmlrpc.server import DocXMLRPCRequestHandler
from xxlimited import new
from cv2 import sqrt
from ents.Element import Element
import cv2
import numpy as np
from math import sin, cos, sqrt, radians

class Agent(Element):
    def __init__(self, name, x_max, x_min, y_max, y_min):
        super(Agent, self).__init__(name, x_max, x_min, y_max, y_min)

        self.icon_w_original = 64
        self.icon_h_original = 64
        self.set_h(64)
        self.set_w(64)
        self.orientation = 0
        self.velocity = np.random.uniform(0, 1)
        try:
            self.icon_original = cv2.imread("/home/wil/BCR/docker-containers/project/ECBeCoCWheR/src/envs/ents/icons/robot.png")
        except Exception as e:
            print(str(e))
        # self.icon = self.icon_original
        # self.icon_h = self.icon_h_original
        # self.icon_w = self.icon_w_original
        self.rotate(np.random.randint(0, 359))
        # self.rotate(-100)

    def get_padding_value(self, angle):
        angle = abs(angle) % 90
        # print("angle ", angle)
        # hyp = sqrt(self.icon_h_original**2 + self.icon_w_original**2)
        # print("hyp ", hyp)
        # print("sin ", abs(sin(radians(angle))))
        # print((hyp) * abs(sin(radians(angle)) / 2))
        padding = (self.icon_h_original/2) * 0.6 * \
                    min(abs(sin(radians(angle))),
                        abs(cos(radians(angle))))
        return int(padding)
    
    def rotate_image(self, turn):
        image = self.icon_original
        self.orientation = (turn + self.get_orientation()) % 360
        padding = self.get_padding_value(self.orientation)
        # print("Orient : ", self.orientation)
        # print("Padding : ", padding)
        image = cv2.copyMakeBorder(self.icon_original, padding, padding, padding, padding, cv2.BORDER_CONSTANT, value=(255,255,255))
        new_image_size = image.shape[0]
        image_center = tuple(np.array(image.shape[1::-1]) / 2)
        matrix = cv2.getRotationMatrix2D(image_center, float(self.orientation), 1)
        self.icon = cv2.warpAffine(image, matrix, (new_image_size, new_image_size), borderValue=(255,255,255))
        self.set_h(new_image_size)
        self.set_w(new_image_size)
        # self.set_h(new_image_size)
        # self.set_w(new_image_size)

    def rotate(self, turn):
        # print("turn: ", turn%360)
        self.rotate_image(turn%360)
    
    def set_velocity(self, nvel):
        self.velocity = nvel
        
        
    def get_orientation(self):
        return self.orientation
    
    def move_agent(self):
        # print("Ori: ", self.orientation)
        # print("Vel : ", self.velocity)
        dx = self.velocity * sin(radians(self.get_orientation()))
        dy = self.velocity * cos(radians(self.get_orientation()))
        # print(-dx, dy)
        self.move(-dx, dy)
        
        # x,y = self.get_position()
        # print(x, y)