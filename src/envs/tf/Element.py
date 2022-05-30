from Point import Point
import cv2

class Element(Point):
    def __init__(self, name, x_max, x_min, y_max, y_min, active_collision = False):
        super(Element, self).__init__(name, x_max, x_min, y_max, y_min)
        self.icon_w = 0
        self.icon_h = 0
        self.active_collision = active_collision

    def activate_collision(self):
        self.active_collision = True
    
    def deactivate_collision(self):
        self.active_collision = False

    def collided(self):
        return self.active_collision
    
    def set_w(self, width):
        self.icon_w = width

    def set_h(self, height):
        self.icon_h = height

    def get_w(self):
        return self.icon_w

    def get_h(self):
        return self.icon_h

        