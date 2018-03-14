from functools import partial
import numpy as np
from scipy.optimize import minimize
from sklearn.linear_model import TheilSenRegressor

EPSILON = 1e-8


class Pareto2:
    def __init__(self, a, b, c, d):
        self.a = a
        self.b = b
        self.c = c
        self.d = d
        self.e = 0.0
        self.f = 0.0

    def update_a_b(self, x, y):
        self.a, self.b = Pareto2.learn_a_b(x, y, (self.c, self.d), (self.e, self.f), a0=self.a, b0=self.b)
        r = np.abs(y-self.get_y(x))
        r[np.abs(r) < EPSILON] = EPSILON
        if np.min(r) < np.min(np.exp(self.c*x+self.d)):
            if self.c == 0.0:
                self.d = np.log(np.min(r)) - EPSILON
            else:
                (self.c, self.d), (self.e, self.f) = Pareto2.get_c_d(x, r)

    def set_c_d(self, c, d):
        self.c = c[0]
        self.d = c[1]
        self.e = d[0]
        self.f = d[1]

    def get_y(self, x):
        return self.a * x + self.b

    def get_likelihood(self, x, y):
        return 1/(2.0*x.shape[0])*np.sum((self.e*x+self.f)+np.exp(self.e*x+self.f)*(self.c*x+self.d)
                                         - (np.exp(self.e*x+self.f)+1)*np.log(np.abs(self.get_y(x)-y)))

    def to_string(self):
        return "a:{}, b:{}, lambda:exp({}*x + {}), alpha:exp({}*x + {})".format(self.a, self.b, self.c, self.d,
                                                                                self.e, self.f)

    def get_a_b(self):
        return self.a, self.b

    @staticmethod
    def var_to_weight(v):
        return np.log(v)

    @staticmethod
    def obj_a_b(l, x, y, lamb, alpha, thresh=EPSILON):
        (c, d) = lamb
        (e, f) = alpha
        obj = 0
        for xi, yi in zip(x, y):
            r = np.abs((l[0]*xi+l[1]-yi))
            w = np.exp(e*xi+f)+1
            if r < thresh:
                obj += np.log(thresh)*w
            else:
                obj += np.log(r)*w
        return obj

    @staticmethod
    def learn_a_b(x, y, lamb, alpha, a0=-0.5, b0=3.4):
        (c, d) = lamb
        (e, f) = alpha
        if (a0 == 0.0) or (b0 == 0.0):
            model = TheilSenRegressor()
            model.fit(x.reshape(-1, 1), y)
            a0 = model.coef_[0]
            b0 = model.intercept_
        if (d == 0) and (c == 0):
            r = a0*x+b0 - y
            d = np.log(np.min(np.abs(r))) - 1e-8

        r = minimize(Pareto2.obj_a_b, [a0, b0], args=(x, y, (c, d), (e, f)), method='Nelder-Mead',
                     options={'maxiter': 10000, 'disp': False})
        # print r
        if not r.success:
            print "Optimization Failed", r
        return r.x[0], r.x[1]

    @staticmethod
    def obj_c_d_e_f(l, x, r):
        c = l[0]
        d = l[1]
        e = l[2]
        f = l[3]
        # print l, np.sum(-(e*x+f)+np.exp(e*x+f)*(np.log(np.abs(r))-(c*x+d))),  np.min(np.log(np.abs(r)) - (l[0]*x+l[1]))
        return np.sum(-(e*x+f)+np.exp(e*x+f)*(np.log(np.abs(r))-(c*x+d)))

    @staticmethod
    def con_r(l, i, r, x):
        return np.log(np.abs(r[i])) - (l[0]*x[i]+l[1])

    @staticmethod
    def get_c_d(x, r, c0=0.0, d0=None, e0=0.0, f0=0.0):
        r[np.abs(r) < EPSILON] = EPSILON
        if d0 is None:
            d0 = np.log(np.min(np.abs(r))) - EPSILON
        constrains = tuple({'type': 'ineq', 'fun': partial(Pareto2.con_r, i=i, r=r, x=x)} for i in xrange(len(r)))
        r = minimize(Pareto2.obj_c_d_e_f, [c0, d0, e0, f0], args=(x, r),
                     options={'maxiter': 10000, 'disp': False}, constraints=constrains)
        # print r
        if not r.success:
            print "Optimization Failed", r
        return (r.x[0], r.x[1]), (r.x[2], r.x[3])

    @staticmethod
    def learn_parameters(x, y, a, b):
        flag = True
        c = 0.0
        d = np.log(np.min(np.abs(y - (a*x + b))))
        e = 0.0
        f = 0.0
        it = 0
        while flag:
            a_p, b_p = a, b
            r = y - (a*x + b)
            print a, b, c, d, 1/(2.0 * x.shape[0]) * np.sum(e*x+f+np.exp(e*x+f)*(c*x+d) -
                                                            (np.exp(e*x+f) + 1)*np.log(np.abs(a*x+b-y)))
            (c, d), (e, f) = Pareto2.get_c_d(x, r)
            a, b = Pareto2.learn_a_b(x, y, (c, d), (e, f))
            flag = (((a-a_p)**2 + (b-b_p)**2) > EPSILON) or it < 10
            it += 1
        p = Pareto2(a, b, c, d)
        p.set_c_d((c, d), (e, f))
        return p
