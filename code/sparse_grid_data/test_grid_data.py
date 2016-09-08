import numpy as np
from pandas import read_csv

nodes = read_csv('points_2d.dat', sep=',', header=None).values
weights = read_csv('weights_2d.dat', sep=',', header=None).values.flatten()

#Compute the error-function

from scipy.integrate import dblquad
from scipy.special import erf #2/rt(pi) * integral( e^{-t**2} )_0^x

integrand = lambda x: 4.0 * np.exp( - np.dot(x,x) ) / np.pi

node_values = np.array( map( integrand, list( nodes ) ))
Q = np.dot( weights, node_values )

print "Computation of integral 4.0 exp(-x^2 - y^2) / pi on unit square"
print Q
print erf(1)**2
print "error = %f" % np.abs( Q - erf(1)**2)

print "Computation of integral 4.0 exp(-x^2-y^2) / pi on [0.25,0.75]^2"
T_nodes = np.zeros_like( nodes )
T_nodes[:,0] = 0.5*nodes[:,0] + 0.25
T_nodes[:,1] = 0.5*nodes[:,1] + 0.25
node_values = np.array( map( integrand, list( T_nodes ) ) )
Q = np.dot( weights, node_values ) * 0.25

print Q
real =  ( erf(0.75) - erf(0.25) )**2
print real
print "error = %f " % np.abs( real - Q ) 
