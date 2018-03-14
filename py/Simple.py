from sklearn.linear_model import TheilSenRegressor

from cvxpy import *
import numpy as np


class Simple:
    def __init__(self, a, b, c, d):
        self.model = TheilSenRegressor()

    def update_a_b(self, x, y):
        self.model.fit(x.reshape(-1, 1), y)

    def set_c_d(self, c, d):
        pass

    def get_y(self, x):
        return self.model.predict(x.reshape(-1, 1))

    def get_likelihood(self, x, y):
        return 1/float(x.shape[0]) * np.sum(np.abs(y-self.get_y(x)))

    def to_string(self):
        return "a:{}, b:{}".format(self.model.coef_, self.model.intercept_)

    def get_a_b(self):
        return self.model.coef_, self.model.intercept_

    @staticmethod
    def var_to_weight(v):
        return 1

    @staticmethod
    def get_c_d(x, r):
        return None, None