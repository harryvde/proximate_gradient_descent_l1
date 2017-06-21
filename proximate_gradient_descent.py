# -*- coding: utf-8 -*-
"""
@author: Harry
"""
from __future__ import division
import numpy as np

def softThreshold(x,lmda):
    """soft thresholding function"""
    return np.sign(x)*np.maximum(abs(x)-lmda,0)

class proximateGradientLasso(object):
    """Class implements gradient Lasso"""
    def __init__(self,model):
        """
        Initialize class
        
        param model likelihoodModel: model to estimate with lasso
        """
        self.model = model
        self.nParam = model.K
        
    def getLoss(self,beta):
        """returns loss"""
        ll = self.model.lossFunction(beta)
        if np.isnan(ll):
            return np.inf
        else:
            return ll

    def getGradient(self,beta):
        """returns loss"""
        return self.model.lossGradient(beta)
       
    def estimate(self, lmda, prec_tol=1e-9, maxIter=1e4 ):
        """run the proximate gradient lasso algorithm"""
        error = np.inf
        beta = np.zeros(self.nParam)
        cpt = 0
        alpha = .5
        while abs(error) > prec_tol and cpt<=maxIter:
            llo = self.getLoss(beta)
            j = self.getGradient(beta)
            error2=True
            t = 1.0
            while error2:
                Gx = (beta - softThreshold(beta - t * j, t*lmda)) / t
                error2 = self.getLoss(beta - t*Gx) > self.getLoss(beta) - t*j.dot(Gx) + t/2.0 * Gx.dot(Gx)
                t*=alpha
            beta = softThreshold(beta - t * j, t*lmda)
            error = self.getLoss(beta) - llo
            llo = error + llo
        return beta

