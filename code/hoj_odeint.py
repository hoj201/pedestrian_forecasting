import numpy as np

def rk4( f , x0 , t_final , N_steps ):
    """ Integrates an ode using 4th order Runge-Kutta 

    args:
        f: (callable) dx/dt = f(x)
        x0: (numpy.array) initial condition
        t_final: (float)
        N_steps: (int) positive

    returns:
        x_arr: (numpy.array) x[k] is x at time t_arr[k]
        t_arr: (numpy.array) 1d arrary of times
    """
    if t_final < 0:
        t_arr = np.linspace( t_final, 0.0, N_steps)
        t_arr = t_arr[::-1]
    else:
        t_arr = np.linspace( 0.0, t_final, N_steps)
    dt = t_arr[1] - t_arr[0]
    x_arr = np.zeros( (N_steps , x0.size ) )
    x_arr[0] = x0
    for k in range(1, N_steps ):
        x = x_arr[k-1]
        k1 = f(x )
        k2 = f(x + k1*dt / 2.0 )
        k3 = f(x + k2*dt / 2.0 )
        k4 = f(x + k3*dt )
        x_arr[k] = x_arr[k-1] + (k1 + 2*k2 + 2*k3 + k4) * dt / 6.0
    return x_arr, t_arr


if __name__ == '__main__':
    f = lambda x: np.array( [x[0],-1.0] )
    print "Test 1:"
    x0 = np.ones(2)
    t_final = 1.0
    N_steps = 100
    x_arr, t_arr = rk4( f , x0, t_final, N_steps )
    x_exact = np.zeros( ( N_steps, 2) )
    x_exact[:,0] = x0[0]*np.exp( np.linspace( 0.0, 1.0, 100 ) )
    x_exact[:,1] = x0[1] -  np.linspace( 0.0, 1.0, 100 )
    print "error = %f \n" %  np.abs( x_arr - x_exact ).sum().max()


    print "Test 2:"
    x_arr, t_arr = rk4( f , x0, -1.0, N_steps )
    x_exact = np.zeros( ( N_steps, 2) )
    x_exact[:,0] = x0[0]*np.exp( t_arr )
    x_exact[:,1] = x0[1] -  t_arr
    print "error = %f" % np.abs( x_arr - x_exact ).sum().max()
