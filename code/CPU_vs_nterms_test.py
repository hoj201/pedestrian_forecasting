import numpy as np
import hermite_function


p_x = np.zeros( (1,1,2,1) )
p_y = np.zeros( (1,1,1,2) )
p_u = np.zeros( (4,5,1,1) )
p_v = np.zeros( (5,4,1,1) )

p_x[0,0,1,0]= 1.
p_y[0,0,0,1]= 1.
M=3

deg=10

sigma = 1.0
mu_x = 0.0
mu_y = 0.0
mu_u = 0.0
mu_v = 0.0
G = lambda x,mu,s : np.exp( - (x-mu)**2 / (2*s**2) )
f = lambda x,y,u,v: G(x,mu_x,sigma)*G(y,mu_y,sigma)*G(u,mu_u,sigma)*G(v,mu_v,sigma)
h_series = hermite_function.hermite_function_series(M=M,deg=deg, dim=4)
h_series.interpolate_function(f)

max_degree = 5
t_arr = np.zeros( [ max_degree ,max_degree] )

for xp in range(0,max_degree):
    for yp in range(0,max_degree):
        # create a polynomial x**xp * y**yp
        c = np.random.rand()+0.1
        if(xp > 0):
            p_u[xp-1,yp,0,0] = -c
        if( yp> 0):
            p_v[xp,yp-1,0,0] = -c
            
        # initialize the EL vector field with this polynomial
        polynomials = [p_x,p_y,p_u,p_v]
        L_f = hermite_function.Lie_derivative( polynomials=polynomials,M=M,deg=deg,dim=4)
        
        # Time a single dot product with h_series
        from time import time
        t0 = time()
        dh_series = L_f.dot( h_series )
        t_arr[xp,yp] = time() - t0
        print 'Computed term  x^%d  y^%d' %(xp,yp)

np.save('CPU_vs_n_terms_data', t_arr )
