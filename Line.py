from fractions import Fraction


# def get_gradient(point_a, point_b):
#     return (point_b.get_y - point_a.get_y) / (point_b.get_x - point_a.get_x)


# def find_equasion_line(point_a, point_b):
# gradient = get_gradient(point_a, point_b)  # get the gradient of the two points
# return point_a, gradient


def find_vertical_line(point_a, point_b):
    gradient = ((point_b.get_x - point_a.get_x) / (point_b.get_y - point_a.get_y)) * -1
    # gradient = Fraction(gradient)
    # gradient = gradient.denominator / gradient.numerator
    x_c = (point_a.get_x + point_b.get_x) / 2
    y_c = (point_a.get_y + point_b.get_y) / 2
    return Point(x_c, y_c), gradient


class Line:
    def __init__(self, point_a, point_b):
        point, gradient = find_vertical_line(point_a, point_b)
        self.point = point
        self.gradient = gradient


class Point:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def get_point(self):
        return self.x, self.y  # x, y = get_point()

    def get_x(self):
        return self.x

    def get_y(self):
        return self.y
