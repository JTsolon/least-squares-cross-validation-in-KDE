# -*- coding: utf-8 -*-

import numpy as np
import statsmodels.api as sm
from scipy.optimize import minimize
from scipy.integrate import quad

'''
This module is designed to find the optimal bandwith under lscv criteria in KDE.
By calling h_cv(sample,a,b), you can get the optimal bandwith that the function return.
'''

#compute the asymptotic squared error, return the cv,squared error minus the integral of raw pdf
def squared_error(h,sample,a,b):
    '''h is a 1-D array
       sample:array_like
       a:lower bound of the support of rv
       b:upper bound of the support of rv'''
       
    h=h[0]
    dens=sm.nonparametric.KDEUnivariate(sample)
    dens.fit(kernel='gau',bw=h)
    f_n_square=lambda x:dens.evaluate(x)[0]**2 #whole sample estimated pdf
    
    #compute the mean of cross prediction value
    def f_sub_sample_mean():
        summation=0
        for i in range(len(sample)):
            subsample=sample.copy()
            del subsample[i]
            dens1=sm.nonparametric.KDEUnivariate(subsample)
            dens1.fit(kernel='gau',bw=h)
            predict_value=dens.evaluate(sample[i])[0]
            summation+=predict_value
        return summation/len(sample)
    
    cv=abs(2*quad(f_n_square,a,b)[0]-2*f_sub_sample_mean()) 
    return cv
    
#find the optimal bandwith h minimizing cv and return it
def h_cv(sample,a,b):
    '''sample:array_like
       a:lower bound of the support of rv
       b:upper bound of the support of rv'''
    h0=[np.array(sample).std()*(len(sample)**(-0.2))] #the initial value of h
    
    #following, h here is constraint to be larger than 10**(-8), otherwise it could be zero when do optimization, 
    #which will raise a 'ZeroDivisionError' exception in method 'evaluate(points)'
    cons=({'type':'ineq','fun':lambda x:x[0]-10**(-8)})
    res=minimize(squared_error,h0,args=(sample,a,b),constraints=cons)
    h=res.x #the optimal h
    #value=res.fun
    
    return h[0]
