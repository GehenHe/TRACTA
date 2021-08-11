#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 3/21/18 5:07 PM
# @Author  : gehen
# @File    : optmize.py
# @Software: PyCharm Community Edition

from numpy.random import randn
import cvxpy as cvx
from qcqp import *

n, m = 10, 15
S = randn(m,m)
x = cvx.Variable(m,n)
obj = cvx.sum_squares(x*x.T-S)
prob = cvx.Problem(cvx.Minimize(obj))
qcqp = QCQP(prob)
qcqp.suggest(SPECTRAL)

f_cd, v_cd = qcqp.improve(COORD_DESCENT)
print("Coordinate descent: objective %.3f, violation %.3f" % (f_cd, v_cd))
print(x.value)