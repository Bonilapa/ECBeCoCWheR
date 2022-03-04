from ents.Point import Point
import cv2

class Bird(Point):
    def __init__(self, name, x_max, x_min, y_max, y_min):
        super(Bird, self).__init__(name, x_max, x_min, y_max, y_min)
        self.icon = cv2.imread("/home/wil/BCR/docker-containers/project/ECBeCoCWheR/src/envs/ents/icons/bird.png")
        self.icon_w = 32
        self.icon_h = 32
        self.icon = cv2.resize(self.icon, (self.icon_h, self.icon_w))
    