from ents.Point import Point
import cv2

class Chopper(Point):
    def __init__(self, name, x_max, x_min, y_max, y_min):
        super(Chopper, self).__init__(name, x_max, x_min, y_max, y_min)
        self.icon_w = 64
        self.icon_h = 64
        try:
            self.icon = cv2.imread("/home/wil/BCR/docker-containers/project/ECBeCoCWheR/src/envs/ents/icons/chopper.png")
            #print(self.icon)
            self.icon = cv2.resize(self.icon, (self.icon_h, self.icon_w))
        except Exception as e:
            print(str(e))


#icon = cv2.imread("/home/wil/BCR/docker-containers/project/ECBeCoCWheR/src/envs/ents/chopper.png")
#print(icon)
#chopper = Chopper("chopper", 50, 10, 50, 10)