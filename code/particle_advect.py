"""
Routines to solve noiseless advection equation.
"""

import numpy as np

def advect( dynamics, nodes, rho_0, t_span ):
    """ Solves the noiseless advection equation when the initial condition is a sum of Dirac-Delta functionals
    
    args:
        dynamics: callable, dynamics( x, jac=False).  Outputs the vector field and possibly the Jacobian
        nodes: ndarray (2,N)
        rho_0: callable, initial density
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

    print "Testing the advection routine in backward time"
    t_span = np.linspace(0,-1,5)
    weights_t = advect( dynamics, nodes, lambda x: rho_0_callable( x[0], x[1] ) , t_span )

    fig, ax_arr = plt.subplots( 1, len(t_span) , figsize = (15,5) )
    for k in range(len(t_span) ):
        weights = weights_t[k].reshape( X_grid.shape) 
        ax_arr[k].contourf( X_grid, Y_grid, weights , cmap = 'viridis' )
        ax_arr[k].set_title( "t= %f" % t_span[k] )
    plt.show()
