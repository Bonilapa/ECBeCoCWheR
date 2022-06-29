from xmlrpc.server import DocXMLRPCRequestHandler
from xxlimited import new
from cv2 import sqrt
from Element import Element
import cv2
import numpy as np
from math import sin, cos, sqrt, radians

from ai import AI
from auv_mpc import KineticModel

class Bird(Element):
    def __init__(self, name, x_max, x_min, y_max, y_min):
        super(Bird, self).__init__(name, x_max, x_min, y_max, y_min)

        self.icon_w_original = 32
        self.icon_h_original = 32
        self.set_h(32)
        self.set_w(32)
        self.set_passive_collision(True)

        try:
            self.icon_original = cv2.imread("/home/wil//ECBeCoCWheR/src/envs/ents/icons/bird.png")
        except Exception as e:
            print(str(e))
        
        # self.icon_original = cv2.resize(self.icon_h_original, (self.icon_h, self.icon_w, 3))
        self.icon = self.icon_original
        # image_center = tuple(np.array(self.icon_original.shape[1::-1]) / 2)
        # matrix = cv2.getRotationMatrix2D(image_center, float(0.0), 1)
        # self.icon = cv2.warpAffine(self.icon_original, matrix, (self.icon_h_original, self.icon_w_original), borderValue=(255,255,255))