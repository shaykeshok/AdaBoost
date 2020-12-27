import Point


def find_gradient(a, b, c, d):
    try:
        return (a - b) / (c - d)
    except ZeroDivisionError:
        return 0


def find_vertical_line(point_a, point_b):
    # print(float(point_b['x']), float(point_a['x']), float(point_b['y']),float(point_a['y']))
    # gradient: float = find_gradient(float(point_b['x']), float(point_a['x']), float(point_b['y']),
    #                                 float(point_a['y'])) * -1
    gradient: float = find_gradient(point_b[0], point_a[0], point_b[1], point_a[1]) * -1
    # x_c = (float(point_a['x']) + float(point_b['x'])) / 2
    # y_c = (float(point_a['y']) + float(point_b['y'])) / 2
    # return x_c, y_c, \
    return gradient


# line (=rule) that separator between two points.
class Line:
    def __init__(self, point_a, point_b):
        # print('point_a:', point_a, ' \npoint_b:', point_b)
        gradient = find_gradient(point_b[0], point_a[0], point_b[1], point_a[1]) * -1
        #     gradient = find_vertical_line(point_a, point_b)
        self.gradient = gradient
        # self.inverse_gradient = find_gradient(float(point_b['y']), float(point_a['y']), float(point_b['x']),
        #                                       float(point_a['x']))
        self.inverse_gradient = find_gradient(point_b[1], point_a[1], point_b[0], point_a[0])
        self.point_a = point_a
        self.point_b = point_b

    # func to check the label of the point in relation to this rule
    def point_labeling(self, point_to_label):
        # side = float(point_to_label['x']) * self.inverse_gradient + (
        #         float(self.point_a['y']) - (self.inverse_gradient * float(self.point_a['x']))) - float(
        #     point_to_label['y'])
        side = point_to_label[0] * self.inverse_gradient + (
                 self.point_a[1] - (self.inverse_gradient * self.point_a[0])) - point_to_label[1]
        return -1 if (side < 0) else 1
