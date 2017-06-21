# -*- coding: utf-8 -*-
"""
Created on Wed Jun 21 22:48:17 2017

@author: Harry
"""

from logistic_regression import logisticRegression
from proximate_gradient_descent import proximateGradientLasso
import numpy as np

N = 1000
Kt = 10
K = 4
e = np.random.randn(N,)
X = np.random.randn(N,Kt)
Xint = np.concatenate((np.ones((N,1)),X[:,0:K]), axis=1)
beta = [1,2,3,4,5]
p = 1.0/(1.0 + np.exp(-Xint.dot(beta))).reshape(N,1)
y = np.random.binomial(1,p)
model = logisticRegression(y,X)
param = model.estimate()

ggl = proximateGradientLasso(model)
pp = ggl.estimate(10)
import pprint
pprint.pprint(np.vstack((pp,param)).T)