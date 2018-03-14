from cvxpy import *
import numpy as np


class Quad:
    def __init__(self, a, b, c, d):
        self.a = a
        self.b = b
        self.c = c
        self.d = d

    def update_a_b(self, x, y):
        self.a, self.b = Quad.learn_a_b(x, y, self.c, self.d)

    def set_c_d(self, c, d):
        self.c = c
        self.d = d

    def get_y(self, x):
        return self.a * x + self.b

    def get_likelihood(self, x, y):
        return -np.log(2*np.pi)/2.0 - 1/(2.0 * x.shape[0]) * np.sum(
            (self.c * x + self.d)+((y-self.get_y(x))**2*np.exp(-(self.c * x + self.d))))

    def to_string(self):
        return "a:{}, b:{}, sigma^2:exp({}*x + {})".format(self.a, self.b, self.c , self.d)

    def get_a_b(self):
        return self.a, self.b

    @staticmethod
    def var_to_weight(v):
        return np.log(v)

    @staticmethod
    def learn_a_b(x, y, c, d):
        w = np.exp(-c*x - d)
        yx = np.sum(x*y*w)
        sx = np.sum(x*w)
        sy = np.sum(y*w)
        sw = np.sum(w)
        sxx = np.sum(x**2*w)
        a = (yx - (sx*sy)/sw)/(sxx - sx**2/sw)
        r = np.sum((y - a*x)*w)
        b = r / sw
        return a, b

    # @staticmethod
    # def get_a_b2(x, y, c, d):
    #     a = Variable(1)
    #     b = Variable(1)
    #     objective = norm(mul_elemwise(exp(-(c * x + d)/2.0), a*x + b - y), 2)
    #     objective = Minimize(objective)
    #     prob = Problem(objective)
    #     prob.solve(solver='SCS', verbose=False)
    #     return a.value, b.value

    @staticmethod
    def get_c_d(x, r):
        c = Variable(1)
        d = Variable(1)
        likelihood = sum_entries(-(c * x + d) - (r ** 2 * exp(-c * x - d)))
        objective = Maximize(likelihood)
        prob = Problem(objective)
        prob.solve(solver='SCS', verbose=False)
        return c.value, d.value

    @staticmethod
    def learn_parameters(x, y, a, b):
        flag = True
        c = 0
        d = np.log(np.sum((y - (a*x + b)) ** 2))
        while flag:
            a_p, b_p = a, b
            r = y - (a*x + b)
            print a, b, c, d, np.sum(-(c * x + d) - (r ** 2 * np.exp(-c * x - d))), np.exp(-c * x - d)
            c, d = Quad.get_c_d(x, r)
            a, b = Quad.learn_a_b(x, y, c, d)
            flag = ((a-a_p)**2 + (b-b_p)**2) > 1e-5

        return Quad(a, b, c, d)