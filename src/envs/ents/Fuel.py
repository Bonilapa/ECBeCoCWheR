from ents.Element import Element
import cv2

class Fuel(Element):
    def __init__(self, name, x_max, x_min, y_max, y_min):
        super(Fuel, self).__init__(name, x_max, x_min, y_max, y_min)
        self.icon = cv2.imread("/home/wil/BCR/docker-containers/project/ECBeCoCWheR/src/envs/ents/icons/fuel.png")
        self.set_h(32)
        self.set_w(32)
        self.icon = cv2.resize(self.icon, (self.get_h(), self.get_w()))