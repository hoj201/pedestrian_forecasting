import numpy as np
import hermite_function

p_x = np.zeros( (1,1,2,1) )
p_y = np.zeros( (1,1,1,2) )
p_u = np.zeros( (4,3,1,1) )
p_v = np.zeros( (5,2,1,1) )

p_x[0,0,1,0]= 1.
p_y[0,0,0,1]= 1.
p_u[3,2,0,0]= -4.
p_v[4,1,0,0]= -2.
polynomials = [p_x,p_y,p_u,p_v]
M=3


sigma = 1.0
mu_x = 0.0
mu_y = 0.0
mu_u = 0.0
mu_v = 0.0
G = lambda x,mu,s : np.exp( - (x-mu)**2 / (2*s**2) )
f = lambda x,y,u,v: G(x,mu_x,sigma)*G(y,mu_y,sigma)*G(u,mu_u,sigma)*G(v,mu_v,sigma)
degree_list = np.arange(5,30)
t_list = np.zeros( degree_list.size)
for i,deg in enumerate(degree_list):
    L_f = hermite_function.Lie_derivative( polynomials=polynomials, M=M, deg=deg , dim=4)
    h_series = hermite_function.hermite_function_series(M=M,deg=deg, dim=4)
    h_series.interpolate_function(f)
    from time import time
    t0 = time()
    new_h_series = L_f.dot( h_series )
    t_list[i] = time()-t0
    print "Completed deg = %d" % deg

np.savez('scaling_by_degree_data' , t_list , degree_list )
