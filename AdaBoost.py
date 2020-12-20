import numpy as np


class AdaBoost:
    def __init__(self, df, hyp, r):
        # self.set_of_hyp = []
        self.df = df
        self.hyp = hyp
        self.r = r

    def active_ada_boost(self):
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
                # Set the weight of the points
                item.weight = item.weight * np.e(-a_t * h_t.include(item, h_t.gender_classifier) * item.gender)
                Z_t += item.weight
            for item in self.df:
                # normalization
                item.weight = item.weight / Z_t
            set_of_hyp.append((h_t, a_t))
            return set_of_hyp
