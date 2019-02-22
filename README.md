# least-squares-cross-validation-in-KDE
This code is to compute the optimal bandwidth based the lscv criteria in kernel density estimation 
The main api of the module in lscv.h_cv(sample,a,b)
parameters:
sample: array-like
a: lower bound of the population, not necessarily accurate
b: upper bound of the population, not necessarily accurate

return: bandwidth, float

