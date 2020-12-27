from itertools import combinations

import numpy as np
from numpy import random
from pandas import DataFrame

from Line import Line


class AdaBoost:

    def __init__(self, df, k, iterations):
        self.df = df
        self.k = k
        self.iterations = iterations

    def active(self):
        for k in range(1, self.k):
            e_𝐻_k_T = 0
            e_𝐻_k_S = 0
            for iterator in range(self.iterations):

                # create 2 random sets from the data: 1-train, 2-test
                random.shuffle(self.df)
                middle = int(self.df.size / 8)
                S = self.df[1:middle]  # Train
                T = self.df[middle + 1:]  # Test

                rules = self.run_adaboost_algo(S, k)

                for point in S:
                    𝐻_k = 0  # 𝐻𝑘(𝑥) = 𝑠𝑖𝑔𝑛(Σ𝛼𝑖ℎ𝑖 (𝑥))
                    for hyp, 𝛼 in rules:
                        𝐻_k += 𝛼 * hyp.point_labeling(point, hyp.c)
                    e_𝐻_k_S += int(𝐻_k * point[2] < 0)
                for point in T:
                    𝐻_k = 0  # 𝐻𝑘(𝑥) = 𝑠𝑖𝑔𝑛(Σ𝛼𝑖ℎ𝑖 (𝑥))
                    for hyp, 𝛼 in rules:
                        𝐻_k += 𝛼 * hyp.point_labeling(point, hyp.c)
                    e_𝐻_k_T += int(𝐻_k * point[2] < 0)

            S_avg = (e_𝐻_k_S / self.iterations) / middle
            T_avg = (e_𝐻_k_T / self.iterations) / middle

            print("\n----------------------- round-" + str(k) + " ------------------------------")
            print("\nAverage error (train):", "%.3f" % S_avg + "%")
            print("\nAverage error (test):", "%.3f" % T_avg + "%")

    def run_adaboost_algo(self, dataset, k):  # dataset:dataframe
        rules_response = []

        # Initialize point weights
        for point in dataset:
            point[3] = 1 / len(dataset)
        for it in range(k):
            𝑍_𝑡 = 0
            ℎ_𝑡, 𝜖_𝑡 = self.find_min_rules(dataset)

            # If 𝜖_𝑡(ℎ_𝑡) is close to ½, weight of classifier is low
            # If 𝜖_𝑡(ℎ_𝑡) is bigger then ½, weight of classifier is worst
            if 𝜖_𝑡 > 0.5 or 𝜖_𝑡 <= 0:
                break

            # Set classifier weight 𝛼_𝑡 based on its error
            𝛼_𝑡 = 0.5 * np.log((1 - 𝜖_𝑡) / 𝜖_𝑡)  # np.log is ln

            # update point weights
            for point in dataset:
                point[3] = point[3] * np.exp(-𝛼_𝑡 * ℎ_𝑡.point_labeling(point, ℎ_𝑡.c) * point[2])
                𝑍_𝑡 += point[3]

            # Normalize these weights
            for point in dataset:
                point[3] = point[3] / 𝑍_𝑡

            rules_response.append((ℎ_𝑡, 𝛼_𝑡))
        return rules_response

    # Select classifier with min weighted error
    def find_min_rules(self, points):
        all_points_combination = combinations(points, 2)

        empirical_error_sum = 1  # min sum of errors
        min_rule = None
        for one_combination in all_points_combination:
            temp_rule = Line(one_combination[0], one_combination[1])
            temp_sum = 0
            temp_right_side = 0
            temp_left_side = 0

            #  Because the line is bi-directional we check here two options of the line:
            #  1. when the line classificated the points on the right side as positive and the points on the left side as negetive
            #  2. when the line classificated the points on the left side as positive and the points on the right side as negetive
            for point in points:
                temp_right_side += point[3] * int(int(point[2]) != temp_rule.point_labeling(point, 1))
                temp_left_side += point[3] * int(int(point[2]) != temp_rule.point_labeling(point, -1))
            temp_sum = temp_right_side
            if temp_left_side < temp_right_side:
                temp_sum = temp_left_side
                temp_rule.c = -1
            if temp_sum < empirical_error_sum:
                empirical_error_sum = temp_sum
                min_rule = temp_rule

        return min_rule, empirical_error_sum
