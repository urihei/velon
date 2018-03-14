from cvxpy import *
import numpy as np


class Pareto1:
    def __init__(self, a, b, c, d):
        self.a = a
        self.b = b
        self.c = c
        self.d = d
        self.alpha = 1

    def update_a_b(self, x, y):
        self.a, self.b = Pareto1.learn_a_b(x, y, self.c, self.d)
        r = np.abs(y-self.get_y(x))

        if np.min(r) < np.min(np.exp(self.c*x+self.d)):
            if self.c == 0.0:
                self.d = np.log(np.min(r)) - 1e-8
            else:
                (self.c, self.d), self.alpha = Pareto1.get_c_d(x, r)

    def set_c_d(self, c, d):
        self.c = c[0]
        self.d = c[1]
        self.alpha = d

    def get_y(self, x):
        return self.a * x + self.b

    def get_likelihood(self, x, y):
        return np.log(self.alpha)/2.0+1/(2.0*x.shape[0])*np.sum(
            self.alpha*(self.c*x+self.d)-(self.alpha + 1)*np.log(np.abs(self.get_y(x)-y)))

    def to_string(self):
        return "a:{}, b:{}, lambda:exp({}*x + {}), alpha:{}".format(self.a, self.b, self.c, self.d, self.alpha)

    def check_validity(self, x, r):
        return ((np.log(np.abs(r)) - (self.c*x + self.d)) < 0).any()

    def get_a_b(self):
        return self.a, self.b

    @staticmethod
    def var_to_weight(v):
        return np.log(v)

    @staticmethod
    def learn_a_b(x, y, c, d):
        a = Variable(1)
        b = Variable(1)
        objective = norm(a*x + b - y, 1)
        objective = Minimize(objective)
        prob = Problem(objective)
        prob.solve(solver='SCS', verbose=False)
        return a.value, b.value

    @staticmethod
    def get_c_d(x, r):
        c = Variable(1)
        d = Variable(1)
        objective = sum_entries(c * x + d)
        objective = Maximize(objective)
        constrain = [c*x[i]+d < log(abs(r[i])) for i in xrange(len(r))]
        prob = Problem(objective, constrain)
        prob.solve(verbose=False)  # solver='SCS',
        alpha = x.shape[0] / np.sum(np.log(np.abs(r)) - (c.value * x + d.value))
        return (c.value, d.value), alpha

    @staticmethod
    def learn_parameters(x, y, a, b):
        flag = True
        c = 0
        d = np.log(np.min(np.abs(y - (a*x + b))))
        alpha = 1
        iter = 0
        while flag:
            a_p, b_p = a, b
            r = y - (a*x + b)
            print a, b, c, d, alpha, 1/(2.0 * x.shape[0]) * np.sum(np.log(alpha) + alpha * (c *x + d)-
                                                            (alpha + 1)*np.log(np.abs(a*x+b-y))),\
                np.exp(c * x + d)
            (c, d), alpha = Pareto1.get_c_d(x, r)
            a, b = Pareto1.learn_a_b(x, y, c, d)
            flag = (((a-a_p)**2 + (b-b_p)**2) > 1e-8) or iter < 10
            iter += 1
        p = Pareto1(a, b, c, d)
        p.set_c_d((c, d), alpha)
        return p