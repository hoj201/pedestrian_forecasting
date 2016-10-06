import numpy as np

def trap_quad( integrand, bounds, res = None ):
    """ Computes integrals using the n-dimensional trapezoid rule

    args:
        integrand: callable
        bounds: iterable of the form (x_min, x_max, y_min, y_max, ... )

    kwargs:
        res: iterable, of the form (x_res, y_res, ...)

    returns:
        q: float

    Notes:
        The integrand should be of the form f(x,y) for a 2D integrand
        where x and y are each 1D numpy arrays of equal length.
    """
    if res is None:
        res = (len(bounds)/2)*(100,)

    if type(res) == int:
        res = (res,)

    assert( len(res)*2 is len(bounds) )
    if len(res) == 1:
        weights = np.ones( res[0] )
        weights[0], weights[-1] = 0.5, 0.5
        grid = np.linspace( bounds[0], bounds[1], res[0] )
        dx = grid[1] - grid[0]
        weights *= dx
        return np.einsum( 'i,i', weights, integrand( grid ) )

    if len(res) == 2:
        weights = np.ones( res )
        weights[0,:] = 0.5
        weights[-1,:] = 0.5
        weights[:,0] = 0.5
        weights[:,-1] = 0.5
        weights[0,0] = 0.25
        weights[0,-1] = 0.25
        weights[-1,0] = 0.25
        weights[-1,1] = 0.25
        x_span = np.linspace( bounds[0], bounds[1], res[0] )
        y_span = np.linspace( bounds[2], bounds[3], res[1] )
        d = lambda x_arr: x_arr[1] - x_arr[0]
        dV = d( x_span) * d(y_span)
        weights *= dV
        X_grid, Y_grid = np.meshgrid( x_span, y_span )
        return np.einsum( 'i,i', weights.flatten(),
                integrand( X_grid, Y_grid ).flatten() ) 

    if len(res) == 3:
        weights = np.ones( res )
        # FACES (there are 6)
        weights[0,:,:], weights[-1,:,:] = 0.5
        weights[:,0,:], weights[:,-1,:] = 0.5
        weights[:,:,0], weights[:,:,-1] = 0.5

        # EDGES (there are 12)
        weights[0,0,:], weights[0,-1,:] = 0.25
        weights[-1,0,:], weights[-1,-1,:] = 0.25
        weights[0,:,0], weights[0,:,-1] = 0.25
        weights[-1,:,0], weights[-1,:,-1] = 0.25
        weights[:,0,0], weights[:,-1,0] = 0.25
        weights[:,-1,0], weights[:,-1,-1] = 0.25

        # CORNERS (there are 8)
        weights[0,0,0], weights[0,0,-1] = 0.125
        weights[0,-1,0], weights[-1,0,0] = 0.125
        weights[0,-1,-1], weights[-1,0,-1] = 0.125
        weights[-1,-1,0], weights[-1,-1,-1] = 0.125

        x_span = np.linspace( bounds[0], bounds[1], res[0] )
        y_span = np.linspace( bounds[2], bounds[3], res[1] )
        z_span = np.linspace( bounds[4], bounds[5], res[2] )
        d = lambda x_arr: x_arr[1] - x_arr[0]
        dV = d( x_span) * d(y_span) * d(z_span)
        weights *= dV
        X_grid, Y_grid, Z_grid = np.meshgrid( x_span, y_span , z_span)
        return np.einsum( 'i,i', weights.flatten(),
                integrand( X_grid, Y_grid, Z_grid ).flatten() ) 
    print "Dimension {dim} is too high for our code.  Aborting.".format(dim=len(res))
    return -1

if __name__ == '__main__':
    integrand = lambda x: x**2
    res = 50
    bounds = (0,1)
    q = trap_quad( integrand, bounds, res )
    print "Testing 1 dimensional quadrature"
    print "result         = {q}".format(q=q)
    print "correct answer = {a}".format(a=1.0/3.0)
    print "error = {error}".format(error=q-1.0/3.0)

    print "\nTesting 2 dimensional quadrature."
    print "Integrand = y(1+x^2)"
    print "Domain = [0.0, 1.0]x[-0.5, 1.0]"
    integrand = lambda x,y: x**2 * y + y
    bounds = (0, 1, -0.5, 1)
    res = (64, 128)
    q = trap_quad( integrand, bounds, res )
    a = (1.0/3.0) * 0.5*( 1.0 - 0.5**2 ) + 0.5*( 1.0 - 0.5**2 )
    print "result         = {q}".format(q=q)
    print "correct answer = {a}".format(a=a)
    print "error = {error}".format(error=q-a)
