from cvxpy import *
import numpy as np


class Exp:
    def __init__(self, a, b, c, d):
        self.a = a
        self.b = b
        self.c = c
        self.d = d

    def update_a_b(self, x, y):
        self.a, self.b = Exp.learn_a_b(x, y, self.c, self.d)

    def set_c_d(self, c, d):
        self.c = c
        self.d = d

    def get_y(self, x):
        return self.a * x + self.b

    def get_likelihood(self, x, y):
        return 1/(2.0 * x.shape[0]) * np.sum(
            (self.c * x + self.d) - (np.abs(y-self.get_y(x)) * np.exp(self.c * x + self.d)))

    def get_a_b(self):
        return self.a, self.b

    def to_string(self):
        return "a:{}, b:{}, lambda:exp({}*x + {})".format(self.a, self.b, self.c, self.d)

    @staticmethod
    def var_to_weight(v):
        return np.log(v)

    @staticmethod
    def learn_a_b(x, y, c, d):
        a = Variable(1)
        b = Variable(1)
        fun = norm(mul_elemwise(exp(c * x + d), a*x + b - y), 1)
        objective = Minimize(fun)
        prob = Problem(objective)
        prob.solve(solver='SCS', verbose=False)
        return a.value, b.value

    @staticmethod
    def get_c_d(x, r):
        c = Variable(1)
        d = Variable(1)
        likelihood = sum_entries((c * x + d) - (mul_elemwise(abs(r), exp(c * x + d))))
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
            print a, b, c, d, 1/(2.0 * x.shape[0]) * np.sum(
                (c * x + d) - (np.abs(y-(a*x+b)) * np.exp(c * x + d))), np.exp(-c * x - d)
            c, d = Exp.get_c_d(x, r)
            a, b = Exp.get_a_b(x, y, c, d)
            flag = ((a-a_p)**2 + (b-b_p)**2) > 1e-8

        return Exp(a, b, c, d)