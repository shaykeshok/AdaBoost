from itertools import combinations

import numpy as np
from numpy import random
from pandas import DataFrame

from Line import Line


class AdaBoost:
    count = 1

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
                # self.df = self.df.sample(frac=1).reset_index(drop=True)
                middle = int(self.df.size / 8)
                S = self.df[1:middle]  # Train
                T = self.df[middle + 1:]  # Test

                rules = self.run_adaboost_algo(S, k)

                # for index, point in S.iterrows():
                for point in S:
                    𝐻_k = 0  # 𝐻𝑘(𝑥) = 𝑠𝑖𝑔𝑛(Σ𝛼𝑖ℎ𝑖 (𝑥))
                    for hyp, 𝛼 in rules:
                        𝐻_k += 𝛼 * hyp.point_labeling(point)
                    # e_𝐻_k_S += int(𝐻_k * point['label'] < 0)
                    e_𝐻_k_S += int(𝐻_k * point[2] < 0)

                # for index, point in T.iterrows():
                for point in T:
                    𝐻_k = 0  # 𝐻𝑘(𝑥) = 𝑠𝑖𝑔𝑛(Σ𝛼𝑖ℎ𝑖 (𝑥))
                    for hyp, 𝛼 in rules:
                        𝐻_k += 𝛼 * hyp.point_labeling(point)
                    # e_𝐻_k_T += int(𝐻_k * point['label'] < 0)
                    e_𝐻_k_T += int(𝐻_k * point[2] < 0)

            S_avg = (e_𝐻_k_S / self.iterations) / middle
            T_avg = (e_𝐻_k_T / self.iterations) / middle

            T_error_string = "\nThe average percentage error on test in round " + str(
                k) + ": " + "%.3f" % T_avg + "(" + "%.3f" % (1 - T_avg) + " % were correct)"
            S_error_string = "\nThe average percentage error on train in round " + str(
                k) + ": " + "%.3f" % S_avg + "(" + "%.3f" % (1 - S_avg) + " % were correct)"
            print(S_error_string)
            print(T_error_string)
            print("\n-----------------------------------------------------")


    def run_adaboost_algo(self, dataset, k):  # dataset:dataframe
        rules_response = []

        # Initialize point weights
        # for point in dataset:
        # for index, row in dataset.iterrows():
        for point in dataset:
            point[3] = 1 / len(dataset)
            # dataset.at[index, 'weight'] = 1 / len(dataset)
        for it in range(k):
            𝑍_𝑡 = 0
            ℎ_𝑡, 𝜖_𝑡 = self.find_min_rules(dataset)

            # If 𝜖_𝑡(ℎ_𝑡) is close to ½, weight of classifier is low
            # If 𝜖_𝑡(ℎ_𝑡) is bigger then ½, weight of classifier is worst
            if 𝜖_𝑡 > 0.5:
                print(self.count, ". 𝜖_𝑡(ℎ_𝑡) is bigger then ½:", 𝜖_𝑡)
                self.count += 1
                break

            # Set classifier weight 𝛼_𝑡 based on its error
            𝛼_𝑡 = 0.5 * np.log((1 - 𝜖_𝑡) / 𝜖_𝑡)  # np.log is ln

            # update point weights
            for point in dataset:
                # point['weight'] = point['weight'] * np.e(-𝛼_𝑡 * ℎ_𝑡.point_labeling(point) * point['label'])
                point[3] = point[3] * np.exp(-𝛼_𝑡 * ℎ_𝑡.point_labeling(point) * point[2])
                # 𝑍_𝑡 += point['weight']
                𝑍_𝑡 += point[3]

            # Normalize these weights
            for point in dataset:
                # point['weight'] = point['weight'] / 𝑍_𝑡
                point[2] = point[2] / 𝑍_𝑡

            rules_response.append((ℎ_𝑡, 𝛼_𝑡))
        return rules_response

    # Select classifier with min weighted error
    # def find_min_rules(self, points: DataFrame):
    def find_min_rules(self, points):
        # all_points_combination = combinations(points.values, 2)
        all_points_combination = combinations(points, 2)

        empirical_error_sum = 1  # min sum of errors
        # min_rule = Line(points[0:1], points[1:2])
        min_rule = Line(points[0], points[1])
        for one_combination in all_points_combination:
            # if one_combination[0].isnumeric() and one_combination[1].isnumeric():
            temp_rule = Line(one_combination[0], one_combination[1])
            temp_sum = 0  # What inside the hypothesis is 1
            # neg_temp_sum = 0  # What inside the hypothesis is -1

            # for index, point in points.iterrows():
            for point in points:
                # temp_sum += point['weight'] * int(point['label'] != temp_rule.point_labeling(point, 1))
                temp_sum += point[3] * int(int(point[2]) != temp_rule.point_labeling(point))
                # neg_temp_sum += point['weight'] * int(point['label'] != temp_rule.point_labeling(point, -1))
                # neg_temp_sum += point[3] * int(point[2] != temp_rule.point_labeling(point))
            # if neg_temp_sum < temp_sum:
            #     temp_sum = neg_temp_sum
            if temp_sum < empirical_error_sum:
                empirical_error_sum = temp_sum
                min_rule = temp_rule

        # all_points_combination = combinations(points.values, 2)
        all_points_combination = combinations(points, 2)
        for one_combination in all_points_combination:
            # if one_combination[0].isnumeric() and one_combination[1].isnumeric():
            temp_rule = Line(one_combination[0], one_combination[1])
            temp_sum = 0  # What inside the hypothesis is 1
            # neg_temp_sum = 0  # What inside the hypothesis is -1

            for point in points:
                # temp_sum += point['weight'] * int(point['label'] != (-1 * temp_rule.point_labeling(point, 1)))
                temp_sum += point[3] * int(point[2] != (-1 * temp_rule.point_labeling(point)))
                # neg_temp_sum += point['weight'] * int(point['label'] != (-1 * temp_rule.point_labeling(point, -1)))
                # neg_temp_sum += point[3] * int(point[2] != (-1 * temp_rule.point_labeling(point)))
            # if neg_temp_sum < temp_sum:
            #     temp_sum = neg_temp_sum
            #     temp_rule.gender_classifier = -1
            if temp_sum < empirical_error_sum:
                empirical_error_sum = temp_sum
                min_rule = temp_rule
        # print(err_sum)
        return min_rule, empirical_error_sum
