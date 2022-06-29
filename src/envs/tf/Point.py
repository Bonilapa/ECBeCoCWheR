import numpy as np
class Point(object):
    def __init__(self, name, x_max, x_min, y_max, y_min):
        self.x = 0.0
        self.y = 0.0
        self.x_min = x_min
        self.x_max = x_max
        self.y_min = y_min
        self.y_max = y_max
        self.name = name
        self.old_x = 0.0
        self.old_y = 0.0
    
    def set_position(self, x, y):
        self.x = self.clamp(x, self.x_min, self.x_max)
        self.y = self.clamp(y, self.y_min, self.y_max)
    
    def get_position(self):
        return (self.x, self.y)
    
    def move(self, del_x, del_y):
        self.x += del_x
        self.y += del_y
        int(np.round(np.float16(self.x)))
        self.x = self.clamp(int(np.round(np.float16(self.x))), self.x_min, self.x_max)
        self.y = self.clamp(int(np.round(np.float16(self.y))), self.y_min, self.y_max)


    def clamp(self, n, minn, maxn):
        return max(min(maxn, n), minn)
