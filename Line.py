def find_gradient(a, b, c, d):
    ans = 0
    if (c - d) == 0:
        return ans
    else:
        ans = (a - b) / (c - d)
    return ans


def find_vertical_line(point_a, point_b):
    gradient: float = find_gradient(point_b[0], point_a[0], point_b[1], point_a[1]) * -1
    return gradient


# line (=rule) that separator between two points.
class Line:
    def __init__(self, point_a, point_b):
        gradient = find_gradient(point_b[0], point_a[0], point_b[1], point_a[1]) * -1
        self.gradient = gradient
        self.inverse_gradient = find_gradient(point_b[1], point_a[1], point_b[0], point_a[0])
        self.point_a = point_a
        self.point_b = point_b
        self.c = 1

    # func to check the label of the point in relation to this rule
    def point_labeling(self, point_to_label, classifier_side):
        side = point_to_label[0] * self.inverse_gradient + (
                self.point_a[1] - (self.inverse_gradient * self.point_a[0])) - point_to_label[1]
        return -1 * classifier_side if (side < 0) else 1 * classifier_side
