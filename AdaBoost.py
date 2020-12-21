import time

import numpy as np
import random


class AdaBoost:
    def __init__(self, df, hyp, r, iterations):  # hyp=Line
        # self.set_of_hyp = []
        self.df = df
        self.hyp = hyp
        self.r = r
        self.iterations = iterations

    def run_adaboost_algo(self, dataset):
        set_of_hyp = []
        for item in self.df:
            item.weight = 1 / len(self.df)
        for t in range(self.r):
            h_t, eps_t = self.hyp(self.df)
            if eps_t > 0.5:
                print("eps too big:", eps_t)
                break
            a_t = 0.5 * np.log((1 - eps_t) / eps_t)  # np.log is ln
            Z_t = 0
            for item in self.df:
                # Set the points weight
                item.weight = item.weight * np.e(-a_t * h_t.include(item, h_t.gender_classifier) * item.gender)
                Z_t += item.weight
            for item in self.df:
                # Normalize these weights
                item.weight = item.weight / Z_t
            set_of_hyp.append((h_t, a_t))
            return set_of_hyp

    def active(self):
        # tic = time.perf_counter()
        print("~~~~~~Line-Rules~~~~~")
        s = "\nNumber of iterations:" + str(self.iterations) + "\nNumber of iterations adaboost:" + str(self.r - 1)

        for r in range(1, self.r):
            sum_total = 0
            S_total = 0
            for i in range(self.iterations):
                random.shuffle(self.df)
                middle = self.df.size() / 2
                S = self.df[0:middle]  # Train
                T = self.df[middle + 1:]  # Test
                res = self.run_adaboost_algo(S)
                for x in T:
                    H_x = 0
                    for hyp, alpha in res:
                        H_x += alpha * hyp.point_labeling(x, hyp.gender_classifier)
                    sum_total += int(H_x * x.gender < 0)
                for x in S:
                    H_x = 0
                    for hyp, alpha in res:
                        H_x += alpha * hyp.point_labeling(x, hyp.gender_classifier)
                    S_total += int(H_x * x.gender < 0)
            T_errors = (sum_total / self.iterations) / 65
            S_errors = (S_total / self.iterations) / 65
            T_error_string = "\nThe average precentage error on T in round " + str(
                r) + ": " + "%.3f" % T_errors + "(" + "%.3f" % (1 - T_errors) + " % were correct)"
            S_error_string = "\nThe average precentage error on R in round " + str(
                r) + ": " + "%.3f" % S_errors + "(" + "%.3f" % (1 - S_errors) + " % were correct)"
            s += S_error_string
            s += T_error_string
            print(S_error_string)
            print(T_error_string)
            s += "\n~~~~~~~~~~~~"
        # toc = time.perf_counter()
        # print("\nthe program took", "%.4f" % (toc - tic), "seconds")
        return s
