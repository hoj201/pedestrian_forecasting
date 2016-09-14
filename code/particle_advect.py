"""
Routines to solve noiseless advection equation.
"""

import numpy as np

def advect( dynamics, nodes, rho_0, t_span ):
    """ Solves the noiseless advection equation when the initial condition is a sum of Dirac-Delta functionals
    
    args:
        dynamics: callable, dynamics( x, jac=False).  Outputs the vector field and possibly the Jacobian
        nodes: ndarray (2,N)
        rho_0: callable, initial density acts on (2,k) array
        t_min: float
        t_max: float
        N_t : int

    return:
        out : ndarray (Nt, N), weights at the requested time.
    """
    N_nodes = nodes.shape[1]
    out = np.zeros( (len(t_span) , N_nodes ))
    for n in range(N_nodes):
        from scipy.integrate import odeint
        xw_initial = np.hstack( [ nodes[:,n] , np.ones(N_nodes) ] )
        xw_t = odeint( dirac_delta_ode, xw_initial, t_span, args=( dynamics , ) )
        x = xw_t[:,:2]
        w = xw_t[:,2]
        out[:,n] = rho_0( x.transpose() )* w
    return out


def advect_vectorized( dynamics, nodes, t_span ):
    """ Solves an advection equation
    
    args:
        dynamics (callable): dynamics( x, jac=False).  Outputs the vector field and possibly the Jacobian for x of shape (2,N)
        nodes (numpy ndarray): shape=(2,N) where N is the number of nodes
        t_span: time for which to solve for

    return:
        x : ndarray (Nt, N), x-values at the requested time.
        y : ndarray (Nt, N), y-values at the requested time.
        w : ndarray (Nt, N), w-values at the requested time.

    Note:  The density at time t is given by rho_0( x[t,k] , y[t,k] )*w[t,k]
    """
    N_nodes = nodes.shape[1]
    out = np.zeros( (len(t_span) , N_nodes ))
    from scipy.integrate import odeint
    state_0 =  np.hstack( [ nodes.flatten() , np.ones(N_nodes) ] )
    state_t = odeint( dirac_delta_ode_vectorized, state_0, t_span, args=( dynamics , ) )
    state_t = state_t.reshape( (len(t_span),3,N_nodes) )
    x = state_t[:,0]
    y = state_t[:,1]
    w = state_t[:,2]
    return x,y,w


def dirac_delta_ode_vectorized( state, t, dynamics ):
    """ Solves for a single Dirac delta.  Formatted for input into scipy.integrate.odeint

    args:
        state (numpy.array): shape=(3*N,) where N = number of nodes
        t (float) : time parameter, not used in this implementation
        dynamics (callable): dynamics( x, jac=True) gives the vector-field and the Jacobian at the reqeuested points

    return:
        out: chance in state
    """
    assert( state.size % 3 == 0 )
    N = state.size / 3
    state = state.reshape( (3,N) )
    x = state[0]
    y = state[1]
    w = state[2]
    X,DX = dynamics( np.vstack([x,y]), jac=True )
    dx = -X[0]
    dy = -X[1]
    dw = -np.einsum('ijj->i', DX)
    dstate = np.vstack( [dx,dy,dw] ).flatten()
    return dstate


def dirac_delta_ode( xw, t , dynamics ):
    """ Solves for a single Dirac delta.  Formatted for input into scipy.integrate.odeint

    args:
        blah blah

    return:
        out
    """
    x = xw[:2]
    w = xw[-1]
    X,DX = dynamics( x , jac=True)
    w_dot = -np.trace( DX )*w
    x_dot = -X
    xw_dot = np.zeros_like(xw)
    xw_dot[:2] = x_dot
    xw_dot[-1] = w_dot
    return xw_dot


if __name__ == '__main__':
    print "Testing dirac_delta_ode routine"
    A = np.array( [ [1.0, -1.0], [1.0, 1.0] ] )
    def dynamics( x, jac=False ):
        dx_dt = np.dot( A , x )
        if jac:
            return dx_dt, A
        return dx_dt
    x_initial = np.array( [1.0, 0.0] )
    w_initial = 2.0

    print "x_initial = " + str( x_initial )
    print "w_initial = " + str( w_initial )

    T = 1.3
    from scipy.linalg import expm
    x_final = np.dot( expm( -T*A ) , x_initial )
    w_final = np.linalg.det( expm(-T*A) ) * w_initial

    xw_initial = np.hstack( [x_initial, w_initial] )
    from scipy.integrate import odeint
    xw_computed = odeint( dirac_delta_ode, xw_initial, [0,T], args=(dynamics, ) )[-1]
    print "computed = " + str( xw_computed)
    print "pen paper= " + str( np.hstack( [x_final, w_final ] ) )


    print "Testing the advection routine in forward time"
    t_span = np.linspace(0,1,5)
    x_span = np.linspace(-1,1,20)
    y_span = np.linspace(-1,1,20)
    X_grid, Y_grid = np.meshgrid( x_span, y_span )
    nodes = np.vstack( [ X_grid.flatten() , Y_grid.flatten() ] )
    sigma = 0.2
    rho_0_callable = lambda X_grid, Y_grid: np.exp( -((X_grid-0.5)**2 + Y_grid**2 ) / (2*sigma**2) ) / np.sqrt( 2*np.pi*sigma**2 )
    weights_0_grid = rho_0_callable( X_grid, Y_grid )
    from matplotlib import pyplot as plt
    plt.contourf( X_grid, Y_grid, weights_0_grid , cmap = 'viridis' )
    plt.show()

    weights_0 = weights_0_grid.flatten()
    weights_t = advect( dynamics, nodes, lambda x: rho_0_callable( x[0], x[1] ) , t_span )

    fig, ax_arr = plt.subplots( 1, len(t_span) , figsize = (15,5) )
    for k in range(len(t_span) ):
        weights = weights_t[k].reshape( X_grid.shape) 
        ax_arr[k].contourf( X_grid, Y_grid, weights , cmap = 'viridis' )
        ax_arr[k].set_title( "t= %f" % t_span[k] )
    plt.show()

    from time import time
    t0 = time()
    print "Testing the advection routine in backward time"
    t_span = np.linspace(0,-1,5)
    weights_t = advect( dynamics, nodes, lambda x: rho_0_callable( x[0], x[1] ) , t_span )
    print "CPU time = %f" % (time() - t0)

    fig, ax_arr = plt.subplots( 1, len(t_span) , figsize = (15,5) )
    for k in range(len(t_span) ):
        weights = weights_t[k].reshape( X_grid.shape) 
        ax_arr[k].contourf( X_grid, Y_grid, weights , cmap = 'viridis' )
        ax_arr[k].set_title( "t= %f" % t_span[k] )
    plt.show()


    print "Testing vectorized routines"
    def dynamics_vectorized( x, jac=False):
        dx_dt = np.dot(A,x)
        if jac:
            return dx_dt, np.tile( A, (x.shape[1], 2 , 2 ) )
        return dx_dt
    t0 = time()
    x,y,w = advect_vectorized( dynamics_vectorized, nodes, t_span)  
    print "CPU time = %f" % (time() - t0 )
    fig, ax_arr = plt.subplots( 1, len(t_span) , figsize = (15,5) )
    for k in range(len(t_span) ):
        rho_grid = rho_0_callable( x[k], y[k] )*w[k]
        rho_grid = rho_grid.reshape( X_grid.shape )
        ax_arr[k].contourf( X_grid, Y_grid, rho_grid , cmap = 'viridis' )
        ax_arr[k].set_title( "t= %f" % t_span[k] )
    plt.show()
