import Point


def find_vertical_line(point_a, point_b):
    gradient = ((point_b.get_x - point_a.get_x) / (point_b.get_y - point_a.get_y)) * -1
    x_c = (point_a.get_x + point_b.get_x) / 2
    y_c = (point_a.get_y + point_b.get_y) / 2
    return Point(x_c, y_c), gradient


# line (=rule) that separator between two points.
class Line:
    def __init__(self, point_a, point_b):
        point, gradient = find_vertical_line(point_a, point_b)
        self.point = point
        self.gradient = gradient

    # func to check the label of the point in relation to this rule
    def point_labeling(self, point_to_label: Point):
        side = point_to_label.get_x() * self.get_a() + self.get_b() - point_to_label.get_y()
        return -1 if (side < 0) else 1


def Line_hyp(points):
    comb = combinations(points, 2)
    err_sum = 1  # min sum of errors
    max_rect = rec_hyp(point(), point())  # the hypothesis
    for tupx in comb:
        rect = rec_hyp(tupx[0], tupx[1])
        temp_sum = 0  # What inside the hypothesis is 1
        neg_temp_sum = 0  # What inside the hypothesis is -1
        for x in points:
            temp_sum += x.weight * int(x.gender != rect.include(x, 1))
            neg_temp_sum += x.weight * int(x.gender != rect.include(x, -1))
        if neg_temp_sum < temp_sum:
            temp_sum = neg_temp_sum
            rect.gender_classifier = -1
        if temp_sum < err_sum:
            err_sum = temp_sum
            max_rect = rect
    # print(err_sum)
    return (max_rect, err_sum)
