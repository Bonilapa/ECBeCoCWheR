\
from cv2 import sqrt
from sympy import Matrix
from Element import Element
import cv2
import numpy as np
from math import sin, cos, sqrt, radians

from ai import AI
from auv_mpc import KineticModel

class Agent(Element):
    def __init__(self, name, x_max, x_min, y_max, y_min, show):
        super(Agent, self).__init__(name, x_max, x_min, y_max, y_min)
        self.message = []
        self.old_x = self.get_position()[0]
        self.old_y = self.get_position()[1]
        self.collected_x = 0.0
        self.collected_y = 0.0
        self.show = show
        self.distances = []
        self.real_position_memory = []
        self.position_memory = []
        self.desired = []
        self.distance = 800
        self.distance_old = 0
        self.accs = 0
        self.icon_w_original = 64
        self.icon_h_original = 64
        self.set_h(64)
        self.set_w(64)
        self.orientation = 0
        self.orientation_old = 0
        self.velocity = np.random.uniform(0, 1)
        self.velocity_old = 0
        # self.signal = np.random.uniform(0, 1)
        try:
            self.icon_original = cv2.imread("/home/wil//ECBeCoCWheR/src/envs/ents/icons/robot.png")
        except Exception as e:
            print(str(e))
        # self.icon = self.icon_original
        # self.icon_h = self.icon_h_original
        # self.icon_w = self.icon_w_original
        self.rotate(np.random.randint(0, 359))
        self.ai = None
        self.kineticModel = KineticModel()
        # self.rotate(-100)

    def agent_reset(self):
        self.set_position(0.0, 0.0)
        self.old_position = self.get_position()
        self.orientation = 0
        self.velocity = np.random.uniform(0, 1)
        self.distance_old = 0
        self.accs = 0
        self.orientation_old = 0
        self.velocity_old = 0
        # self.signal = np.random.uniform(0, 1)
        self.rotate(np.random.randint(0, 359))

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
        if padding < 0.0:
            padding = 0
        return int(padding)
    
    def rotate_image(self):
        image = self.icon_original
        padding = self.get_padding_value(self.orientation)
        # print("Orient : ", self.orientation)
        # print("Padding : ", padding)
        image = cv2.copyMakeBorder(self.icon_original, padding, padding, padding, padding, cv2.BORDER_CONSTANT, value=(255,255,255))

        # print(image.shape)
        new_image_size = image.shape[0]
        image_center = tuple(np.array(image.shape[1::-1]) / 2)
        matrix = cv2.getRotationMatrix2D(image_center, float(self.orientation), 1)
        # print(matrix)
        self.icon = cv2.warpAffine(image, matrix, (new_image_size, new_image_size), borderValue=(255,255,255))
        # print(self.icon.shape)
        self.set_h(new_image_size)
        self.set_w(new_image_size)
        # self.set_h(new_image_size)
        # self.set_w(new_image_size)

    def rotate(self, turn):
        # print("turn: ", turn%360)
        self.orientation = (turn + self.get_orientation()) % 360
        if self.show:
            self.rotate_image()
    
    def set_velocity(self, nvel):
        self.velocity = nvel

    # def set_signal(self, sig):
    #     self.signal = sig
    # def get_signal(self):
    #     return self.signal
        
        
    def get_orientation(self):
        return self.orientation
    
    def move_agent(self):
        # print("Ori: ", self.orientation)
        # print("Vel : ", self.velocity)
        dx = self.collected_x + self.velocity * sin(radians(self.get_orientation()))
        dy = self.collected_y + self.velocity * cos(radians(self.get_orientation()))
        # print(-dx, dy)
        self.move(dx, dy)
        # new_x, new_y = self.get_position()

        # print("\n", int(np.round(np.float16(self.pos_hat[0]))), int(np.round(np.float16(self.pos_hat[1]))), \
        #         "__", int(np.round(np.float16(-dx))), int(np.round(np.float16(dy))), "\n")
        # if new_x == self.old_x and not new_x == self.x_min and not new_x == self.x_max:
        #     self.collected_x += dx
        # else:
        #     self.old_x = new_x
        #     self.collected_x = 0.0

        # if new_y == self.old_y and not new_y == self.y_min and not new_y == self.y_max:
        #     self.collected_y += dy
        # else:
        #     self.old_y = new_y
        #     self.collected_y = 0.0

        
        # x,y = self.get_position()
        # print(x, y)
    def accs_compute(self):
        accs = []
        # ddx     
        accs.append((self.velocity - self.velocity_old) * sin(self.orientation))
        # ddy
        accs.append((self.velocity - self.velocity_old)* cos(self.orientation))
        # ddz     
        accs.append(0)
        # ddphi   
        accs.append(0)
        # ddth
        accs.append(0)
        # ddpsi   
        accs.append(self.orientation - self.orientation_old)
        return accs
        
    def get_distance(self):
        return self.distance

    def get_distance_old(self):
        return self.distance_old

    def get_agent_velocities(self, control_values):
        
        accs = self.accs_compute()

        self.orientation_old = self.orientation
        self.velocity_old = self.velocity
        
        vels = self.kineticModel.get_velocities(accs, control_values, self.orientation)
        self.pos_hat = self.kineticModel.get_position(self.orientation)
        # print(np.array(self.pos_hat).shape)
        
        self.position_memory.append(self.pos_hat)
        self.real_position_memory.append(self.get_position())
        # int(np.round(np.float16(self.x)))
        # self.distance = int(np.round(np.float16(sqrt((self.pos_hat[0] + self.get_position()[0] - bird.get_position()[0])**2
        #                     +
        #                     (self.pos_hat[1] + self.get_position()[1] - bird.get_position()[1])**2))))
        # self.distances.append(self.distance)
        return vels

    def choose_action(self, observation):
        a, t, p, v = self.ai.choose_action(observation)
        return a, t, p, v

    def gen_message(self, observation):
        a, t, p, v = self.ai.gen_message(observation)
        return a, t, p, v
