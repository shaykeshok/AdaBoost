class Point:
    def __init__(self, x, y, label, weight=0):
        self.x = x
        self.y = y
        self.label = int(label)
        self.weight = weight

    def get_point(self):
        return self.x, self.y  # x, y = get_point()

    def get_x(self):
        return self.x

    def get_y(self):
        return self.y

    def get_label(self):
        return self.label

    def get_weight(self):
        return self.weight

    def set_weight(self, new_weight):
        self.weight = new_weight
